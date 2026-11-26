"""Readers and binary layouts for ACCORD's live PLY point cloud."""
from pathlib import Path

import numpy as np


def read_active_ply(path):
    """Read ACCORD's binary little-endian XYZ/RGB PLY into float32 points and uint8 colours."""
    with Path(path).open('rb') as stream:
        properties = []
        vertex_count = None
        is_vertex = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'Truncated PLY header: {path}')
            fields = line.decode('ascii', 'strict').split()
            if not fields or fields[0] in ('comment', 'obj_info'):
                continue
            if fields[0] == 'format':
                if fields[1:3] != ['binary_little_endian', '1.0']:
                    raise ValueError(f'Unsupported active PLY format in {path}')
            elif fields[0] == 'element':
                is_vertex = fields[1] == 'vertex'
                if is_vertex:
                    vertex_count = int(fields[2])
            elif fields[0] == 'property' and is_vertex:
                if len(fields) != 3:
                    raise ValueError(f'Unsupported active PLY property in {path}')
                properties.append((fields[1], fields[2]))
            elif fields[0] == 'end_header':
                break

        expected = [('float', 'x'), ('float', 'y'), ('float', 'z'),
                    ('uchar', 'red'), ('uchar', 'green'), ('uchar', 'blue')]
        if vertex_count is None or properties != expected:
            raise ValueError(f'Unexpected active PLY vertex layout in {path}')
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ])
        records = np.fromfile(stream, dtype=dtype, count=vertex_count)
        if len(records) != vertex_count:
            raise ValueError(f'Truncated active PLY vertices in {path}')
    points = np.column_stack((records['x'], records['y'], records['z']))
    colors = np.column_stack((records['red'], records['green'], records['blue']))
    return points, colors


def pack_xyzrgb(points, colors):
    """Pack xyz and RGB into the conventional 16-byte PointCloud2 xyz/rgb layout."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError('Point and colour arrays must both have shape (N, 3)')
    packed = np.empty(len(points), dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4'),
    ]))
    packed['x'], packed['y'], packed['z'] = points.T
    rgb_bits = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2]
    packed['rgb'] = rgb_bits.view(np.float32)
    return packed
