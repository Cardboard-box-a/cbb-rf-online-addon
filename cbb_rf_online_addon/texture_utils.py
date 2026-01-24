import os
import subprocess
import tempfile
from typing import Optional, Tuple, List, Dict
import bpy
import numpy as np
import struct
from enum import Enum
from PIL import Image

HAS_TEXTURE2DDECODER = False

try:
    import texture2ddecoder
    HAS_TEXTURE2DDECODER = True
except ImportError:
    HAS_TEXTURE2DDECODER = False
    print("Warning: texture2ddecoder not available, using slower manual DXT decoding")

def decode_dxt3_alpha_manual(dds_data: bytes, width: int, height: int) -> np.ndarray:
    """
    Manual DXT3 alpha decoding (fallback when texture2ddecoder is unavailable).
    """
    block_width = (width + 3) // 4
    block_height = (height + 3) // 4
    alpha_channel = np.zeros((height, width), dtype=np.uint8)
    offset = 128
    
    for block_y in range(block_height):
        for block_x in range(block_width):
            if offset + 16 > len(dds_data):
                break
            
            alpha_block = struct.unpack('<Q', dds_data[offset:offset + 8])[0]
            offset += 16
            
            for py in range(4):
                for px in range(4):
                    pixel_x = block_x * 4 + px
                    pixel_y = block_y * 4 + py
                    
                    if pixel_x >= width or pixel_y >= height:
                        continue
                    
                    bit_index = (py * 4 + px) * 4
                    alpha_4bit = (alpha_block >> bit_index) & 0xF
                    alpha_8bit = (alpha_4bit * 17)  # Fast multiplication instead of (alpha_4bit * 255) // 15
                    alpha_channel[pixel_y, pixel_x] = alpha_8bit
    
    return alpha_channel


def decode_dxt5_alpha_manual(dds_data: bytes, width: int, height: int) -> np.ndarray:
    """
    Manual DXT5 alpha decoding (fallback when texture2ddecoder is unavailable).
    """
    block_width = (width + 3) // 4
    block_height = (height + 3) // 4
    alpha_channel = np.zeros((height, width), dtype=np.uint8)
    offset = 128
    
    for block_y in range(block_height):
        for block_x in range(block_width):
            if offset + 16 > len(dds_data):
                break
            
            alpha0 = dds_data[offset]
            alpha1 = dds_data[offset + 1]
            indices = struct.unpack('<Q', dds_data[offset:offset + 8])[0] >> 16
            offset += 16
            
            # Build alpha palette
            palette = [alpha0, alpha1]
            if alpha0 > alpha1:
                for i in range(1, 7):
                    palette.append(((7 - i) * alpha0 + i * alpha1) // 7)
            else:
                for i in range(1, 5):
                    palette.append(((5 - i) * alpha0 + i * alpha1) // 5)
                palette.append(0)
                palette.append(255)
            
            for py in range(4):
                for px in range(4):
                    pixel_x = block_x * 4 + px
                    pixel_y = block_y * 4 + py
                    
                    if pixel_x >= width or pixel_y >= height:
                        continue
                    
                    pixel_index = py * 4 + px
                    alpha_index = (indices >> (pixel_index * 3)) & 0x7
                    alpha_channel[pixel_y, pixel_x] = palette[alpha_index]
    
    return alpha_channel


def analyze_dds_alpha(dds_path: str) -> Dict[str, any]:
    """
    Analyze DDS alpha channel and recommend transparency mode.
    Uses texture2ddecoder for DXT5 only, manual decoding for DXT3.
    """
    try:
        with open(dds_path, 'rb') as f:
            dds_data = f.read()
        
        if dds_data[:4] != b'DDS ':
            raise ValueError("Not a valid DDS file")
        
        # Read header
        height = struct.unpack('<I', dds_data[12:16])[0]
        width = struct.unpack('<I', dds_data[16:20])[0]
        pf_flags = struct.unpack('<I', dds_data[80:84])[0]
        fourcc = dds_data[84:88]
        
        has_alpha_flag = bool(pf_flags & 0x1)
        
        print(f"  DDS Analysis: {width}x{height}, FourCC: {fourcc}, AlphaFlag: {has_alpha_flag}")
        
        # Handle DXT1 without alpha
        if fourcc == b'DXT1' and not has_alpha_flag:
            return {
                'has_alpha': False,
                'mode': 'OPAQUE',
                'threshold': None,
                'histogram': {},
                'binary_percentage': 100.0
            }
        
        # Decode alpha channel
        if fourcc == b'DXT3':
            # DXT3 / BC2 - NOT SUPPORTED by texture2ddecoder, we use manual decoding
            print(f"  Using manual Python decoding for DXT3 (texture2ddecoder doesn't support BC2)")
            alpha = decode_dxt3_alpha_manual(dds_data, width, height)
            
        elif fourcc == b'DXT5':
            # DXT5 / BC3 - Use fast decoder if available
            if HAS_TEXTURE2DDECODER:
                try:
                    rgba = texture2ddecoder.decode_bc3(dds_data[128:], width, height)
                    alpha = np.frombuffer(rgba, dtype=np.uint8)[3::4].reshape(height, width)
                    print(f"  Used texture2ddecoder (fast C++ path) for DXT5")
                except Exception as e:
                    print(f"  texture2ddecoder failed, falling back to manual: {e}")
                    alpha = decode_dxt5_alpha_manual(dds_data, width, height)
            else:
                print(f"  Using manual Python decoding for DXT5")
                alpha = decode_dxt5_alpha_manual(dds_data, width, height)
                
        elif fourcc == b'DXT1':
            # DXT1 with alpha - Use fast decoder if available
            if HAS_TEXTURE2DDECODER:
                try:
                    rgba = texture2ddecoder.decode_bc1(dds_data[128:], width, height)
                    alpha = np.frombuffer(rgba, dtype=np.uint8)[3::4].reshape(height, width)
                    print(f"  Used texture2ddecoder (fast C++ path) for DXT1")
                except Exception as e:
                    print(f"  texture2ddecoder failed: {e}")
                    # DXT1 alpha is too complex to decode manually, assume needs masking
                    return {
                        'has_alpha': True,
                        'mode': 'MASK',
                        'threshold': 0.5,
                        'histogram': {},
                        'binary_percentage': 50.0
                    }
            else:
                # Conservative fallback
                return {
                    'has_alpha': True,
                    'mode': 'MASK',
                    'threshold': 0.5,
                    'histogram': {},
                    'binary_percentage': 50.0
                }
        else:
            raise ValueError(f"Unsupported format: {fourcc}")
        
        # Build histogram efficiently with numpy
        unique, counts = np.unique(alpha, return_counts=True)
        histogram = dict(zip(unique.tolist(), counts.tolist()))
        
        # Calculate binary percentage
        binary_pixels = histogram.get(0, 0) + histogram.get(255, 0)
        total_pixels = width * height
        binary_percentage = (binary_pixels / total_pixels) * 100
        
        print(f"  Alpha values found: {sorted(histogram.keys())[:20]}...")
        print(f"  Binary percentage: {binary_percentage:.1f}%")
        
        # Determine mode
        if binary_percentage > 98:
            mode = 'MASK'
            non_binary = [v for v in histogram.keys() if 0 < v < 255]
            threshold = min(non_binary) / 255.0 if non_binary else 0.5
        elif binary_percentage > 90:
            mode = 'MASK'
            threshold = 0.1
        else:
            mode = 'BLEND'
            threshold = None
        
        return {
            'has_alpha': True,
            'mode': mode,
            'threshold': threshold,
            'histogram': histogram,
            'binary_percentage': binary_percentage
        }
        
    except Exception as e:
        print(f"Error analyzing DDS alpha: {e}")
        import traceback
        traceback.print_exc()
        return {
            'has_alpha': True,
            'mode': 'BLEND',
            'threshold': 0.5,
            'histogram': {},
            'binary_percentage': 0.0
        }



class TextureProcessingError(Exception):
    """Custom exception for texture processing errors"""
    pass

def check_imagemagick() -> bool:
    """Check if ImageMagick is installed and accessible"""
    try:
        from wand.image import Image
        with Image() as img:
            return True
    except Exception:
        return False

            
def ensure_dependencies() -> bool:
    if not check_imagemagick():
        raise TextureProcessingError(
            "ImageMagick is not installed or not accessible.\n"
            "Please install ImageMagick from https://imagemagick.org/script/download.php\n"
            "Make sure to check 'Install legacy utilities' during installation."
        )

    return True
    

def get_dxt_format(image: bpy.types.Image) -> str:
    """Determine the best DXT format for the image"""
    # Check if image has alpha
    has_alpha = image.channels == 4
    
    # Check if it's a normal map (based on name convention)
    is_normal = any(x in image.name.lower() for x in ['normal', 'nrm', 'norm'])
    
    if is_normal:
        return 'dxt5'  # Best for normal maps
    elif has_alpha:
        return 'dxt3'  # For images with alpha
    else:
        return 'dxt1'  # For RGB images

def convert_to_dds(image: bpy.types.Image) -> Optional[bytes]:
    """Convert an image to DDS format with appropriate compression"""
    from wand.image import Image
    import numpy as np
    
    if image.size[0] == 0 or image.size[1] == 0:
        raise TextureProcessingError(f"Invalid image size for {image.name}")
    
    # Create temporary files for conversion
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in, \
         tempfile.NamedTemporaryFile(suffix='.dds', delete=False) as temp_out:
        
        # Get image data
        pixels = np.array(image.pixels[:])
        width, height = image.size
        rgba = (pixels.reshape(height, width, 4) * 255).astype(np.uint8)
        
        # Save as PNG first (Wand works better with files)
        with Image.from_array(rgba) as img:
            img.save(filename=temp_in.name)
        
        # Convert to DDS using ImageMagick command-line
        dxt_format = get_dxt_format(image)
        convert_cmd = [
            'magick',
            'convert',
            temp_in.name,
            '-quality', '100',
            '-depth', '8',
            '-flip',
            '-define',
            f'dds:compression={dxt_format}',
            '-define',
            'dds:mipmaps=6',  # Disable mipmaps
            temp_out.name
        ]
        
        try:
            subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"ImageMagick conversion failed: {e.stderr}")
            return None
        
        # Read the resulting DDS file
        try:
            with open(temp_out.name, 'rb') as f:
                dds_data = f.read()
        except IOError as e:
            print(f"Failed to read converted DDS file: {e}")
            return None
        finally:
            # Cleanup temporary files
            try:
                os.unlink(temp_in.name)
                os.unlink(temp_out.name)
            except OSError:
                pass
        
        return dds_data

class D3DFormat(Enum):
    R5G6B5 = "D3DFMT_R5G6B5"
    A8R8G8B8 = "D3DFMT_A8R8G8B8"

def convert_to_dds_with_format(image: bpy.types.Image, format: D3DFormat = D3DFormat.R5G6B5) -> Optional[bytes]:
    """
    Convert an image to DDS format with specified compression.
    Supports D3DFMT_R5G6B5 and D3DFMT_A8R8G8B8.
    
    :param image: The Blender image to convert.
    :param format: The target format (D3DFMT_R5G6B5 or D3DFMT_A8R8G8B8).
    :return: The DDS file data as bytes.
    """
    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image size for {image.name}")
    
    width, height = image.size
    pixels = np.array(image.pixels[:])
    rgba = (pixels.reshape(height, width, 4) * 255).astype(np.uint8)

    # Flip the Y-axis of the image
    rgba = rgba[::-1, ...]

    
    if format == D3DFormat.R5G6B5:
        # Convert RGBA to RGB565
        r = (rgba[..., 0] >> 3).astype(np.uint16) << 11
        g = (rgba[..., 1] >> 2).astype(np.uint16) << 5
        b = (rgba[..., 2] >> 3).astype(np.uint16)
        rgb565 = (r | g | b).flatten().tobytes()
    elif format == D3DFormat.A8R8G8B8:
        # Convert RGBA to A8R8G8B8 (direct mapping)
        argb = np.zeros((height, width, 4), dtype=np.uint8)
        argb[..., 0] = rgba[..., 3]  # Alpha
        argb[..., 1] = rgba[..., 0]  # Red
        argb[..., 2] = rgba[..., 1]  # Green
        argb[..., 3] = rgba[..., 2]  # Blue
        rgb565 = argb.flatten().tobytes()
    else:
        raise ValueError(f"Unsupported format: {format}")

    dds_pixel_format = (
        32,                      # Size of Pixel Format
        0x40 if format == D3DFormat.R5G6B5 else 0x1,  # Flags (DDPF_RGB | DDPF_ALPHAPIXELS)
        b'\0\0\0\0' if format == D3DFormat.R5G6B5 else b'8888',  # FourCC for uncompressed
        16 if format == D3DFormat.R5G6B5 else 32,      # RGBBitCount
        0xF800 if format == D3DFormat.R5G6B5 else 0xFF0000,  # RBitMask
        0x07E0 if format == D3DFormat.R5G6B5 else 0xFF00,    # GBitMask
        0x001F if format == D3DFormat.R5G6B5 else 0xFF,      # BBitMask
        0x0000 if format == D3DFormat.R5G6B5 else 0xFF000000  # AlphaBitMask (only for A8R8G8B8)
    )
    dds_pixel_format_packed = struct.pack('<2I4s5I', *dds_pixel_format)
    print(f"Managed to pack the dds pixel format, error is after.")
    # Create DDS header
    dds_header = struct.pack(
        '<4s18I32s5I',
        b'DDS ',                     # Magic number
        124,                         # Size of header
        0xA1007,                     # Flags (DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT)
        height,                      # Height
        width,                       # Width
        width * height // (2 if format == D3DFormat.R5G6B5 else 4),  # Pitch or linear size
        0,                           # Depth
        1,                           # MipMapCount
        *[0] * 11,                   # Reserved
        dds_pixel_format_packed,                           # Pixel Format
        0x401008,                      # Caps (DDSCAPS_TEXTURE)
        0, 0, 0,                      # Reserved Caps
        0
    )

    # Combine header and pixel data
    dds_data = dds_header + rgb565

    return dds_data

def convert_bytes_to_dds(image_data: bytes, width: int, height: int, dxt_format: str = "dxt1") -> Optional[bytes]:
    """Convert raw image bytes to DDS format with appropriate compression."""
    from wand.image import Image
    import numpy as np

    if width == 0 or height == 0:
        raise TextureProcessingError(f"Invalid image dimensions: {width}x{height}")
    
    # Create temporary files for conversion
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in, \
         tempfile.NamedTemporaryFile(suffix='.dds', delete=False) as temp_out:
        
        # Convert raw image bytes to PNG format using PIL or numpy
        try:
            rgba = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)
        except ValueError as e:
            raise TextureProcessingError(f"Invalid image data format: {e}")
        
        # Save as PNG first (ImageMagick works better with files)
        with Image.from_array(rgba) as img:
            img.save(filename=temp_in.name)
        
        # Convert to DDS using ImageMagick command-line
        convert_cmd = [
            'magick',
            'convert',
            temp_in.name,
            '-quality', '100',
            '-depth', '8',
            '-flip',  # Flip vertically to handle orientation issues
            '-define',
            f'dds:compression={dxt_format}',
            '-define',
            'dds:mipmaps=6',  # Generate mipmaps
            temp_out.name
        ]
        
        try:
            subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"ImageMagick conversion failed: {e.stderr}")
            return None
        
        # Read the resulting DDS file
        try:
            with open(temp_out.name, 'rb') as f:
                dds_data = f.read()
        except IOError as e:
            print(f"Failed to read converted DDS file: {e}")
            return None
        finally:
            # Cleanup temporary files
            try:
                os.unlink(temp_in.name)
                os.unlink(temp_out.name)
            except OSError:
                pass
        
        return dds_data
