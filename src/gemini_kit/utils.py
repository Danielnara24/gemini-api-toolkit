import requests
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io
import os
import logging
import mimetypes
import pathlib
import hashlib
import base64
import time
import numpy as np
import itertools
import google.genai as genai
from google.genai import types
from typing import Any, List, Optional, Union, Dict

# Initialize logger for this module
logger = logging.getLogger(__name__)

def _add_citations(response: types.GenerateContentResponse) -> str:
    """
    Processes a Gemini response to add inline citations and a formatted source list.
    """
    try:
        metadata = response.candidates[0].grounding_metadata
        if not metadata:
            return response.text
    except (IndexError, AttributeError):
        return response.text

    text = response.text
    supports = metadata.grounding_supports
    chunks = metadata.grounding_chunks

    if not supports or not chunks:
        return text

    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        chunk_indices = sorted(list(set(support.grounding_chunk_indices)))

        if chunk_indices:
            citation_links = []
            for i in chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = "".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    if chunks:
        source_list_header = "\n\n---\n**Sources:**\n"
        source_list = []
        for i, chunk in enumerate(chunks):
            title = chunk.web.title or "Source"
            uri = chunk.web.uri
            source_list.append(f"{i + 1}. [{title}]({uri})")

        text += source_list_header + "\n".join(source_list)

    return text

def _parse_video_timestamp(value: Union[str, int, float, None]) -> Optional[str]:
    """
    Parses timestamps (int, 'MM:SS', 'HH:MM:SS') into '123s' format.
    """
    if value is None:
        return None
    
    # If it's already a number, return as seconds string
    if isinstance(value, (int, float)):
        return f"{int(value)}s"
    
    val_str = str(value).strip()
    
    # If it contains colon, parse as time format
    if ":" in val_str:
        try:
            parts = val_str.split(":")
            seconds = 0
            for part in parts:
                seconds = seconds * 60 + float(part)
            return f"{int(seconds)}s"
        except ValueError:
            return None

    # If it ends with 's', assume valid (e.g. "40s")
    if val_str.lower().endswith("s"):
        return val_str
    
    # If string is just a number
    if val_str.isdigit():
        return f"{val_str}s"
        
    return None

def _get_remote_file_name(client: genai.Client, file_path: str) -> str | None:
    """
    Checks if a local file is already uploaded to Gemini by comparing SHA-256 hashes.
    
    Args:
        client: The initialized genai.Client object.
        file_path: The local path to the file.
        
    Returns:
        str: The remote file name (e.g., 'files/abc123xyz') if found.
        None: If the file is not found on the server.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    except FileNotFoundError:
        return None

    # Google stores the Hex Digest encoded in Base64
    local_hash_b64 = base64.b64encode(sha256_hash.hexdigest().encode('utf-8')).decode('utf-8')

    try:
        # iterate through remote files looking for a match
        for remote_file in client.files.list():
            if remote_file.sha256_hash == local_hash_b64:
                return remote_file.name
    except Exception:
        # If listing fails (e.g. permission or network), assume not found
        return None
            
    return None

def _process_media_attachments(
    client: genai.Client, 
    media_paths: List[Union[str, Dict[str, Any]]], 
    inline_limit_mb: float = 20.0,
    media_resolution: Optional[Dict[str, str]] = None
) -> Union[List[types.Part], str]:
    """
    Processes media files (Local paths, YouTube URLs, and YouTube Dicts with timestamps).
    Supports Images, Videos, Audio, and PDFs.
    """
    if not media_paths:
        return []

    final_parts = []
    
    # Candidates for local processing: (path, size_bytes, mime_type)
    local_candidates = []
    
    for item in media_paths:
        if not item:
            continue
        
        # --- 1. Normalize Input (Distinguish between URL/Dict and Local Path) ---
        target_path_or_url = ""
        video_meta = None
        
        if isinstance(item, dict):
            target_path_or_url = item.get("url", "")
            # Process timestamps if present
            start = _parse_video_timestamp(item.get("start"))
            end = _parse_video_timestamp(item.get("end"))
            
            if start or end:
                video_meta = types.VideoMetadata(
                    start_offset=start,
                    end_offset=end
                )
        else:
            target_path_or_url = str(item)

        # --- 2. Check for YouTube / Web URL ---
        # Simple check for YouTube URLs to handle them separately from local files
        if "youtube.com" in target_path_or_url or "youtu.be" in target_path_or_url:
            part_args = {
                "file_data": types.FileData(file_uri=target_path_or_url),
                "media_resolution": media_resolution
            }
            if video_meta:
                part_args["video_metadata"] = video_meta
                
            final_parts.append(types.Part(**part_args))
            continue
            
        # --- 3. Local File Processing ---
        path = target_path_or_url
        file_path = pathlib.Path(path)
        
        if not file_path.exists():
            msg = f"Error: File not found at '{path}'"
            logger.error(msg)
            return msg

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            msg = f"Error: Could not determine MIME type for '{path}'."
            logger.error(msg)
            return msg
        
        # Supported types
        is_supported = (
            mime_type.startswith("image/") or 
            mime_type.startswith("video/") or 
            mime_type.startswith("audio/") or 
            mime_type == "application/pdf"
        )
        
        if not is_supported:
             msg = f"Error: Unsupported file type '{mime_type}' for file '{path}'."
             logger.error(msg)
             return msg

        # Check if already uploaded
        remote_name = _get_remote_file_name(client, path)
        if remote_name:
            logger.info(f"File '{path}' found remotely as '{remote_name}'. Using existing file.")
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{remote_name}"
            final_parts.append(types.Part(
                file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
                media_resolution=media_resolution
            ))
        else:
            # It's a candidate for local logic (inline vs upload)
            file_size = file_path.stat().st_size
            local_candidates.append({
                "path": path,
                "size": file_size,
                "mime": mime_type,
                "path_obj": file_path
            })

    if not local_candidates:
        return final_parts

    # --- 4. Optimization Logic (Local Files Only) ---
    limit_bytes = inline_limit_mb * 1024 * 1024
    inline_files = []
    upload_files = []

    total_candidate_size = sum(c["size"] for c in local_candidates)

    if total_candidate_size <= limit_bytes:
        inline_files = local_candidates
    else:
        # Greedy fallback for speed if many files, otherwise combinations
        if len(local_candidates) > 20:
            sorted_candidates = sorted(local_candidates, key=lambda x: x["size"])
            current_sum = 0
            for c in sorted_candidates:
                if current_sum + c["size"] <= limit_bytes:
                    inline_files.append(c)
                    current_sum += c["size"]
                else:
                    upload_files.append(c)
        else:
            best_sum = 0
            best_combination = []
            indices = range(len(local_candidates))
            for r in range(len(local_candidates) + 1):
                for subset_indices in itertools.combinations(indices, r):
                    current_subset = [local_candidates[i] for i in subset_indices]
                    current_size = sum(x["size"] for x in current_subset)
                    if current_size <= limit_bytes:
                        if current_size > best_sum:
                            best_sum = current_size
                            best_combination = current_subset
            
            inline_files = best_combination
            inline_paths = set(x["path"] for x in inline_files)
            upload_files = [x for x in local_candidates if x["path"] not in inline_paths]

    # --- 5. Process Inline Files ---
    for item in inline_files:
        try:
            file_bytes = item["path_obj"].read_bytes()
            final_parts.append(types.Part(
                inline_data=types.Blob(data=file_bytes, mime_type=item["mime"]),
                media_resolution=media_resolution
            ))
        except Exception as e:
            msg = f"Error reading file '{item['path']}': {e}"
            logger.error(msg)
            return msg

    # --- 6. Process Upload Files ---
    for item in upload_files:
        logger.info(f"Uploading '{item['path']}' ({item['size']/1024/1024:.2f} MB)...")
        try:
            uploaded_file = client.files.upload(file=item["path"])
            
            # Wait for ACTIVE state
            while True:
                myfile = client.files.get(name=uploaded_file.name)
                if myfile.state.name == "ACTIVE":
                    break
                elif myfile.state.name == "FAILED":
                    msg = f"Error: File processing failed for '{item['path']}' on Google's side."
                    logger.error(msg)
                    return msg
                time.sleep(1)
            
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{myfile.name}"
            
            final_parts.append(types.Part(
                file_data=types.FileData(file_uri=file_uri, mime_type=item["mime"]),
                media_resolution=media_resolution
            ))
            logger.info(f"Upload complete: {item['path']}")
            
        except Exception as e:
            msg = f"Error uploading file '{item['path']}': {e}"
            logger.error(msg)
            return msg

    return final_parts

# --- Helper Function to Clean JSON Markdown ---

def _clean_json_markdown(text: str) -> str:
    """Removes markdown code fencing from JSON strings."""
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
    return text.strip()

# --- Helper Function for 2D Detection ---

def _visualize_2d(image_path_or_url: str, bounding_boxes: List[Dict[str, Any]]) -> Image.Image:
    """
    Draws bounding boxes and labels on the image. 
    Handles both local paths and URLs for the source image.
    """
    try:
        # Load Image
        if str(image_path_or_url).startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            im = Image.open(io.BytesIO(response.content))
        else:
            im = Image.open(image_path_or_url)
        
        # Ensure image is in a mode that supports color drawing
        if im.mode != 'RGB':
            im = im.convert('RGB')

        draw = ImageDraw.Draw(im)
        width, height = im.size
        
        # distinct colors for different objects
        colors = [
            'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
            'cyan', 'magenta', 'lime', 'teal', 'coral'
        ]

        # Use default font or try to load a better one if available
        try:
            # Try loading a standard font, fallback to default
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for i, box_data in enumerate(bounding_boxes):
            box = box_data.get("box_2d")
            label = box_data.get("label", "Object")
            
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            
            abs_y1 = int(ymin / 1000 * height)
            abs_x1 = int(xmin / 1000 * width)
            abs_y2 = int(ymax / 1000 * height)
            abs_x2 = int(xmax / 1000 * width)

            # Draw Rectangle
            color = colors[i % len(colors)]
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)
            
            # Draw Label with background
            text_bbox = draw.textbbox((abs_x1, abs_y1), label, font=font)
            # offset label slightly above or inside box
            text_loc = (abs_x1, max(0, abs_y1 - 20))
            
            # optional: draw text background for readability
            draw.rectangle(draw.textbbox(text_loc, label, font=font), fill=color)
            draw.text(text_loc, label, fill="black" if color in ['yellow', 'lime', 'cyan'] else "white", font=font)

        return im
    except Exception as e:
        logger.error(f"Failed to visualize detection: {e}")
        return None

# --- Helper Functions for Segmentation Visualization ---

def _visualize_segmentation(image_path_or_url: str, json_data: List[Dict[str, Any]]) -> Optional[Image.Image]:
    """
    Overlays all segmentation masks, bounding boxes, and labels on a single image.
    """
    try:
        # Load Image
        if str(image_path_or_url).startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            im = Image.open(io.BytesIO(response.content))
        else:
            im = Image.open(image_path_or_url)

        if im.mode != 'RGBA':
            im = im.convert('RGBA')

        width, height = im.size
        
        # Colors for cycling
        colors = [
            'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
            'cyan', 'magenta', 'lime', 'teal', 'coral'
        ]
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        # Text layer
        text_layer = Image.new('RGBA', im.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_layer)

        for i, item in enumerate(json_data):
            box = item.get("box_2d")
            mask_b64 = item.get("mask")
            label = item.get("label", "Object")
            
            if not box or not mask_b64:
                continue

            ymin, xmin, ymax, xmax = box
            abs_y0 = int(ymin / 1000 * height)
            abs_x0 = int(xmin / 1000 * width)
            abs_y1 = int(ymax / 1000 * height)
            abs_x1 = int(xmax / 1000 * width)

            if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
                continue

            color_name = colors[i % len(colors)]
            try:
                rgb_color = ImageColor.getrgb(color_name)
            except:
                rgb_color = (255, 0, 0)

            try:
                # Handle Mask
                if "base64," in mask_b64:
                    mask_b64 = mask_b64.split("base64,")[1]
                
                mask_data = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_data))
                
                # Resize
                mask_img = mask_img.resize((abs_x1 - abs_x0, abs_y1 - abs_y0), Image.Resampling.BILINEAR)
                mask_arr = np.array(mask_img)
                
                # Overlay
                object_overlay = np.zeros((abs_y1 - abs_y0, abs_x1 - abs_x0, 4), dtype=np.uint8)
                mask_indices = mask_arr > 127
                object_overlay[mask_indices] = rgb_color + (160,) # Color + Alpha
                
                overlay_pil = Image.fromarray(object_overlay, 'RGBA')
                im.alpha_composite(overlay_pil, (abs_x0, abs_y0))

            except Exception as e:
                logger.warning(f"Failed to process mask for {label}: {e}")

            # Draw Box and Label
            draw.rectangle(((abs_x0, abs_y0), (abs_x1, abs_y1)), outline=color_name, width=3)
            text_loc = (abs_x0, max(0, abs_y0 - 20))
            draw.rectangle(draw.textbbox(text_loc, label, font=font), fill=color_name)
            draw.text(text_loc, label, fill="black" if color_name in ['yellow', 'lime', 'cyan'] else "white", font=font)

        final_image = Image.alpha_composite(im, text_layer)
        return final_image.convert('RGB')

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return None

def _save_segmentation_artifacts(image_path_or_url: str, json_data: List[Dict[str, Any]], output_dir: str):
    """
    Saves individual masks and overlay images for each detected item to the output directory.
    """
    try:
        # Load Image
        if str(image_path_or_url).startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            im = Image.open(io.BytesIO(response.content))
        else:
            im = Image.open(image_path_or_url)

        if im.mode != 'RGBA':
            im = im.convert('RGBA')

        os.makedirs(output_dir, exist_ok=True)
        width, height = im.size

        for i, item in enumerate(json_data):
            label = item.get("label", "unknown").replace(" ", "_")
            box = item.get("box_2d")
            mask_b64 = item.get("mask")

            if not box or not mask_b64:
                continue

            # Coordinates
            ymin, xmin, ymax, xmax = box
            y0 = int(ymin / 1000 * height)
            x0 = int(xmin / 1000 * width)
            y1 = int(ymax / 1000 * height)
            x1 = int(xmax / 1000 * width)

            if y0 >= y1 or x0 >= x1:
                continue

            # Process Mask
            try:
                if "base64," in mask_b64:
                    mask_b64 = mask_b64.split("base64,")[1]
                
                mask_data = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_data))
                
                # Resize mask to bounding box
                mask_img = mask_img.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
                mask_array = np.array(mask_img)

                # 1. Save raw mask
                mask_filename = f"{label}_{i}_mask.png"
                mask_img.save(os.path.join(output_dir, mask_filename))

                # 2. Create Overlay (White, Alpha=200)
                # Using numpy for speed instead of pixel loop
                overlay_arr = np.zeros((height, width, 4), dtype=np.uint8)
                
                # Create the specific cutout for the bbox
                cutout = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
                
                # Apply white color (255, 255, 255, 200) where mask > 128
                threshold_indices = mask_array > 128
                cutout[threshold_indices] = (255, 255, 255, 200)
                
                # Place cutout into full size overlay
                overlay_arr[y0:y1, x0:x1] = cutout
                
                overlay_pil = Image.fromarray(overlay_arr, 'RGBA')
                
                # Composite
                composite = Image.alpha_composite(im, overlay_pil)
                
                # Save Overlay
                overlay_filename = f"{label}_{i}_overlay.png"
                composite.save(os.path.join(output_dir, overlay_filename))
                
                logger.info(f"Saved mask and overlay for {label} to {output_dir}")

            except Exception as e:
                logger.warning(f"Failed to save artifact for {label}: {e}")

    except Exception as e:
        logger.error(f"Failed to save segmentation artifacts: {e}")

# --- Helper Function for Pointing Visualization ---

def _visualize_points(image_path_or_url: str, json_data: List[Dict[str, Any]]) -> Optional[Image.Image]:
    """
    Draws points and labels. Handles both standard pointing and multiview (in_frame check).
    """
    try:
        # Load Image
        if str(image_path_or_url).startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            im = Image.open(io.BytesIO(response.content))
        else:
            im = Image.open(image_path_or_url)

        if im.mode != 'RGB':
            im = im.convert('RGB')

        draw = ImageDraw.Draw(im)
        width, height = im.size
        
        colors = ['red', 'lime', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        main_color = "#2962FF" 
        point_radius = 5

        for i, item in enumerate(json_data):
            # MULTIVIEW HANDLER: Check if item is explicitly marked out of frame
            if item.get("in_frame") is False:
                continue

            point = item.get("point")
            label = item.get("label", "Point")
            
            if not point or len(point) != 2:
                continue

            norm_y, norm_x = point
            abs_y = int(norm_y / 1000 * height)
            abs_x = int(norm_x / 1000 * width)

            # 1. Draw Point 
            # Outer white ring (outline)
            draw.ellipse((abs_x - point_radius - 2, abs_y - point_radius - 2, 
                          abs_x + point_radius + 2, abs_y + point_radius + 2), fill="white")
            # Inner blue circle
            draw.ellipse((abs_x - point_radius, abs_y - point_radius, 
                          abs_x + point_radius, abs_y + point_radius), fill=main_color)
            
            # 2. Draw Label
            # Offset text slightly to the right
            text_loc = (abs_x + 15, abs_y - 10)
            
            # Simple bounds check to keep label on screen
            if text_loc[0] > width - 80: 
                # If too far right, move to the left of the dot
                text_width = draw.textlength(label, font=font)
                text_loc = (abs_x - text_width - 15, abs_y - 10)

            # Get text bounding box
            left, top, right, bottom = draw.textbbox(text_loc, label, font=font)
            
            # Draw blue background rectangle with padding
            padding = 4
            draw.rectangle((left - padding, top - padding, right + padding, bottom + padding), 
                           fill=main_color)
            
            # Draw white text
            draw.text(text_loc, label, fill="white", font=font)

        return im

    except Exception as e:
        logger.error(f"Pointing visualization failed: {e}")
        return None
