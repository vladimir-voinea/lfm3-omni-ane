# Minimal PNG codec (8-bit RGB/RGBA, non-interlaced) so the vision path needs no extra
# dependency: the server should not require Pillow just to accept an image.
import struct
import zlib

import numpy as np


def png_decode(data: bytes) -> np.ndarray:
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
    pos, width, height, depth, colour = 8, 0, 0, 0, 0
    payload = b''
    while pos < len(data):
        length, kind = struct.unpack('>I4s', data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b'IHDR':
            width, height, depth, colour = struct.unpack('>IIBB', chunk[:10])
        elif kind == b'IDAT':
            payload += chunk
        elif kind == b'IEND':
            break
        pos += 12 + length
    assert depth == 8 and colour in (2, 6), 'only 8-bit RGB/RGBA PNG'
    channels = 3 if colour == 2 else 4
    raw = zlib.decompress(payload)
    stride = width * channels
    out = np.zeros((height, stride), np.uint8)
    prior = np.zeros(stride, np.uint8)
    at = 0
    for row in range(height):
        filter_type = raw[at]
        at += 1
        line = np.frombuffer(raw[at:at + stride], np.uint8).astype(np.int32)
        at += stride
        if filter_type == 0:
            current = line
        elif filter_type == 1:
            current = line.copy()
            for i in range(channels, stride):
                current[i] = (current[i] + current[i - channels]) & 0xFF
        elif filter_type == 2:
            current = (line + prior) & 0xFF
        elif filter_type == 3:
            current = line.copy()
            for i in range(stride):
                left = current[i - channels] if i >= channels else 0
                current[i] = (current[i] + ((left + int(prior[i])) >> 1)) & 0xFF
        elif filter_type == 4:
            current = line.copy()
            for i in range(stride):
                left = current[i - channels] if i >= channels else 0
                up = int(prior[i])
                corner = int(prior[i - channels]) if i >= channels else 0
                estimate = left + up - corner
                da, db, dc = abs(estimate - left), abs(estimate - up), abs(estimate - corner)
                nearest = left if (da <= db and da <= dc) else (up if db <= dc else corner)
                current[i] = (current[i] + nearest) & 0xFF
        else:
            raise ValueError('bad PNG filter %d' % filter_type)
        out[row] = current.astype(np.uint8)
        prior = current.astype(np.uint8)
    pixels = out.reshape(height, width, channels)
    return pixels[:, :, :3]


def png_encode(pixels: np.ndarray) -> bytes:
    height, width, _ = pixels.shape
    raw = b''.join(b'\x00' + pixels[row].tobytes() for row in range(height))

    def chunk(kind, body):
        return (struct.pack('>I', len(body)) + kind + body
                + struct.pack('>I', zlib.crc32(kind + body) & 0xFFFFFFFF))

    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))