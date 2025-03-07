use std::fs::File;

use byteorder::{LittleEndian, ReadBytesExt};

#[repr(C)]
pub struct HeaderFMSH {
    pub file_magic: u32,
    pub n_submeshes: u32,
    pub offset_mesh_desc: u32,
    pub offset_vertex_data: u32,
    pub offset_mesh_names: u32,
    pub offset_lightmap_uv: u32,
    pub offset_lightmap_tex: u32,
}

#[repr(C)]
pub struct MeshDesc {
    pub vertex_start: u16, // First vertex index for this model
    pub n_triangles: u16,  // Number of vertices for this model
    pub n_quads: u16,      // Number of vertices for this model
    pub x_min: i16,        // Axis aligned bounding box minimum X
    pub x_max: i16,        // Axis aligned bounding box maximum X
    pub y_min: i16,        // Axis aligned bounding box minimum Y
    pub y_max: i16,        // Axis aligned bounding box maximum Y
    pub z_min: i16,        // Axis aligned bounding box minimum Z
    pub z_max: i16,        // Axis aligned bounding box maximum Z
    pub pad: i16,
}

#[repr(C)]
pub struct VertexPSX {
    pub x: i16,    // Position X
    pub y: i16,    // Position Y
    pub z: i16,    // Position Z
    pub r: u8,     // Color R
    pub g: u8,     // Color G
    pub b: u8,     // Color B
    pub u: u8,     // Texture Coordinate U
    pub v: u8,     // Texture Coordinate V
    pub extra: u8, // In the first vertex, this is an index into the texture collection, which determines which texture to use. In the second vertex, this is the size of the triangle.
}

#[derive(Debug)]
#[repr(C)]
pub struct Vertex {
    pub x: i16,
    pub y: i16,
    pub z: i16,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub section_id: u16,
}
impl Vertex {
    pub fn position(&self) -> glam::Vec3 {
        return glam::Vec3::new(self.x as f32, self.y as f32, self.z as f32);
    }
}

pub struct VertexCol {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl HeaderFMSH {
    pub fn read(buf_reader: &mut File) -> HeaderFMSH {
        HeaderFMSH {
            file_magic: buf_reader.read_u32::<LittleEndian>().unwrap(),
            n_submeshes: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_mesh_desc: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_vertex_data: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_mesh_names: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_lightmap_uv: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_lightmap_tex: buf_reader.read_u32::<LittleEndian>().unwrap(),
        }
    }
}

impl MeshDesc {
    pub fn read(buf_reader: &mut File) -> MeshDesc {
        MeshDesc {
            vertex_start: buf_reader.read_u16::<LittleEndian>().unwrap(),
            n_triangles: buf_reader.read_u16::<LittleEndian>().unwrap(),
            n_quads: buf_reader.read_u16::<LittleEndian>().unwrap(),
            x_min: buf_reader.read_i16::<LittleEndian>().unwrap(),
            x_max: buf_reader.read_i16::<LittleEndian>().unwrap(),
            y_min: buf_reader.read_i16::<LittleEndian>().unwrap(),
            y_max: buf_reader.read_i16::<LittleEndian>().unwrap(),
            z_min: buf_reader.read_i16::<LittleEndian>().unwrap(),
            z_max: buf_reader.read_i16::<LittleEndian>().unwrap(),
            pad: 0,
        }
    }
}

impl VertexPSX {
    pub fn read(buf_reader: &mut File) -> VertexPSX {
        VertexPSX {
            x: buf_reader.read_i16::<LittleEndian>().unwrap(),
            y: buf_reader.read_i16::<LittleEndian>().unwrap(),
            z: buf_reader.read_i16::<LittleEndian>().unwrap(),
            r: buf_reader.read_u8().unwrap(),
            g: buf_reader.read_u8().unwrap(),
            b: buf_reader.read_u8().unwrap(),
            u: buf_reader.read_u8().unwrap(),
            v: buf_reader.read_u8().unwrap(),
            extra: buf_reader.read_u8().unwrap(),
        }
    }
}

impl VertexCol {
    pub fn read(buf_reader: &mut File) -> VertexCol {
        VertexCol {
            x: buf_reader.read_i32::<LittleEndian>().unwrap(),
            y: buf_reader.read_i32::<LittleEndian>().unwrap(),
            z: buf_reader.read_i32::<LittleEndian>().unwrap(),
        }
    }
}
