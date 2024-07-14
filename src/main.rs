use byteorder::{LittleEndian, ReadBytesExt};
mod structs;
use renderer::Renderer;
use structs::*;
mod renderer;
mod bvh;
use bvh::*;
mod aabb;

use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    mem::size_of,
    path::Path, str::FromStr,
};

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if (args.len() != 4)
        || (!args[1].ends_with(".msh"))
        || (!args[2].ends_with(".col"))
        || (!args[3].ends_with(".vis"))
    {
        // println!("Usage: psx_vislist_generator.exe <input.msh> <input.col> <output.vis>");
        // return;
        args.push(String::from_str("level.msh").unwrap());
        args.push(String::from_str("level.col").unwrap());
        args.push(String::from_str("level.vis").unwrap());
    }

    let mut input_col = File::open(Path::new(args[2].as_str())).unwrap();
    let visbox_positions = get_visbox_positions_from_input_col(&mut input_col).unwrap();
    drop(input_col);

    let mut rnd_pos_bvh = Bvh::new(visbox_positions);
    rnd_pos_bvh.pad_bounding_boxes(250.0, 250.0, 250.0);
    
    let mut input_msh = File::open(Path::new(args[1].as_str())).unwrap();
    let (_mesh_descs, vertices) = get_mesh_descs_from_input_msh(&mut input_msh).unwrap();
    drop(input_msh);

    let mut renderer = Renderer::new();
    renderer.upload_mesh_to_gpu(&vertices);

    // Traverse the BVH
    let mut vislist_bitfields = Vec::<u128>::new();
    let mut vislist_nodes = Vec::<PsxNode>::new();
    for node in &rnd_pos_bvh.nodes {
        vislist_nodes.push(PsxNode {
            min_x: node.bounds.min.x as i16,
            min_y: node.bounds.min.y as i16,
            min_z: node.bounds.min.z as i16,
            max_x: node.bounds.max.x as i16,
            max_y: node.bounds.max.y as i16,
            max_z: node.bounds.max.z as i16,
            child_or_vis_index: node.left_first as u32,
        })
    }
    let mut node_stack = Vec::<usize>::new();
    node_stack.push(0);

    while let Some(node_id) = node_stack.pop() {
        let node = rnd_pos_bvh.nodes.get(node_id).expect("Node index out of bounds! BVH must be invalid.");
        match node.count {
            0 => { // internal node
                node_stack.push(node.left_first as usize);
                node_stack.push(node.left_first as usize + 1);
            },
            _ => { // leaf node
                let start = node.left_first;
                let end = node.left_first + node.count;
                let mut visbits: u128 = 0u128;
                let leaf_id = vislist_bitfields.len();
                println!("from node {node_id}");
                for i in start..end {
                    if rnd_pos_bvh.vertices[rnd_pos_bvh.indices[i as usize] as usize].b == 0 {
                        continue
                    }
                    let position = rnd_pos_bvh.vertices[rnd_pos_bvh.indices[i as usize] as usize].position();
                    visbits |= renderer.get_visibility_at_position(position, visbits, leaf_id);
                }
                println!("");
                vislist_nodes[node_id].child_or_vis_index = (1 << 31) | (vislist_bitfields.len() as u32);
                vislist_bitfields.push(visbits)
            }
        }
    }

    // Get offsets to data
    let offset_vis_bvh = 0u32;
    let offset_vis_lists = offset_vis_bvh + vislist_nodes.len() as u32 * std::mem::size_of::<PsxNode>() as u32;

    // Open output file that will store the visibility lists
    let mut output_vis = match File::create(Path::new(args[3].as_str())) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to open file: {err}");
            return;
        }
    };

    // Write the file
    output_vis.write_all(&0x53495646u32.to_le_bytes()).unwrap();
    output_vis.write_all(&offset_vis_bvh.to_le_bytes()).unwrap();
    output_vis.write_all(&offset_vis_lists.to_le_bytes()).unwrap();
    for node in vislist_nodes {
        output_vis.write_all(&node.min_x.to_le_bytes()).unwrap();
        output_vis.write_all(&node.min_y.to_le_bytes()).unwrap();
        output_vis.write_all(&node.min_z.to_le_bytes()).unwrap();
        output_vis.write_all(&node.max_x.to_le_bytes()).unwrap();
        output_vis.write_all(&node.max_y.to_le_bytes()).unwrap();
        output_vis.write_all(&node.max_z.to_le_bytes()).unwrap();
        output_vis.write_all(&node.child_or_vis_index.to_le_bytes()).unwrap();
    }
    for vl in vislist_bitfields{
        output_vis.write_all(&vl.to_le_bytes()).unwrap();
    }
}

#[derive(Debug)]
struct PsxNode {
    min_x: i16,
    min_y: i16,
    min_z: i16,
    max_x: i16,
    max_y: i16,
    max_z: i16,
    child_or_vis_index: u32,
}

fn get_visbox_positions_from_input_col(
    input_col: &mut File,
) -> Option<Vec<Vertex>> {
    let mut render_positions: Vec<Vertex> = Vec::new();
    let eye_height = 200;
    let jump_height = eye_height + 100;

    // Make sure it is a valid file
    let file_magic = input_col.read_u32::<LittleEndian>().unwrap();
    if file_magic != 0x4C4F4346 {
        eprintln!("Failed to open collision model!");
        return None;
    }
    
    // Find triangle data
    let n_tris = input_col.read_u32::<LittleEndian>().unwrap() / 3;
    input_col.seek(SeekFrom::Start(12)).unwrap();
    let triangle_data_offset = input_col.read_u32::<LittleEndian>().unwrap();
    
    // Loop over all triangles
    for _ in 0..n_tris {
        // Read vertex
        let v0 = VertexCol::read(input_col);
        let v1 = VertexCol::read(input_col);
        let v2 = VertexCol::read(input_col);
        let normal = VertexCol::read(input_col);
        let normal = glam::vec3(normal.x as f32, normal.y as f32, normal.z as f32);

        render_positions.push(Vertex {
            x: (((v0.x as f32) + (v1.x as f32) + (v2.x as f32)) / 3.0) as i16,
            y: (((v0.y as f32) + (v1.y as f32) + (v2.y as f32)) / 3.0) as i16 - eye_height,
            z: (((v0.z as f32) + (v1.z as f32) + (v2.z as f32)) / 3.0) as i16,
            r: 0,
            g: 0,
            b: match normal.y > 0.5 {
                true => 1,
                false => 0,
            },
            section_id: 0,
        });

        render_positions.push(Vertex {
            x: (((v0.x as f32) + (v1.x as f32) + (v2.x as f32)) / 3.0) as i16,
            y: (((v0.y as f32) + (v1.y as f32) + (v2.y as f32)) / 3.0) as i16 - jump_height,
            z: (((v0.z as f32) + (v1.z as f32) + (v2.z as f32)) / 3.0) as i16,
            r: 0,
            g: 0,
            b: match normal.y > 0.5 {
                true => 1,
                false => 0,
            },
            section_id: 0,
        });
    }
    Some(render_positions)
}

fn get_mesh_descs_from_input_msh(input_msh: &mut File) -> Option<(Vec<MeshDesc>, Vec<Vertex>)> {
    // Read header
    let header_fmsh = HeaderFMSH::read(input_msh);

    // Check file magic
    if header_fmsh.file_magic != 0x48534D46 {
        eprintln!("File identifier invalid! File is either not an FMSH or corrupted");
        return None;
    }

    // Get all the mesh descriptions
    let mut mesh_descs = Vec::new();
    for i in 0..header_fmsh.n_submeshes {
        input_msh
            .seek(SeekFrom::Start(
                size_of::<HeaderFMSH>() as u64
                    + header_fmsh.offset_mesh_desc as u64
                    + (size_of::<MeshDesc>() * i as usize) as u64,
            ))
            .unwrap();
        mesh_descs.push(MeshDesc::read(input_msh));
    }

    // Read all the vertex buffers
    let mut vertices: Vec<Vertex> = Vec::new();
    for (mesh_id, mesh) in mesh_descs.iter().enumerate() {
        // Seek to vertex data
        input_msh
            .seek(SeekFrom::Start(
                size_of::<HeaderFMSH>() as u64
                    + header_fmsh.offset_vertex_data as u64
                    + (mesh.vertex_start as u64 * size_of::<VertexPSX>() as u64),
            ))
            .unwrap();

        // Read all the triangles
        for _ in 0..mesh.n_triangles {
            let v0 = VertexPSX::read(input_msh);
            let v1 = VertexPSX::read(input_msh);
            let v2 = VertexPSX::read(input_msh);
            vertices.push(Vertex {
                x: v0.x,
                y: v0.y,
                z: v0.z,
                r: v0.r,
                g: v0.g,
                b: v0.b,
                section_id: mesh_id as u16,
            });
            vertices.push(Vertex {
                x: v1.x,
                y: v1.y,
                z: v1.z,
                r: v1.r,
                g: v1.g,
                b: v1.b,
                section_id: mesh_id as u16,
            });
            vertices.push(Vertex {
                x: v2.x,
                y: v2.y,
                z: v2.z,
                r: v2.r,
                g: v2.g,
                b: v2.b,
                section_id: mesh_id as u16,
            });
        }

        // Read all the quads and convert them to triangles
        for _ in 0..mesh.n_quads {
            let v0 = VertexPSX::read(input_msh);
            let v1 = VertexPSX::read(input_msh);
            let v2 = VertexPSX::read(input_msh);
            let v3 = VertexPSX::read(input_msh);

            // Tri 1
            vertices.push(Vertex {
                x: v0.x,
                y: v0.y,
                z: v0.z,
                r: v0.r,
                g: v0.g,
                b: v0.b,
                section_id: mesh_id as u16,
            });
            vertices.push(Vertex {
                x: v1.x,
                y: v1.y,
                z: v1.z,
                r: v1.r,
                g: v1.g,
                b: v1.b,
                section_id: mesh_id as u16,
            });
            vertices.push(Vertex {
                x: v2.x,
                y: v2.y,
                z: v2.z,
                r: v2.r,
                g: v2.g,
                b: v2.b,
                section_id: mesh_id as u16,
            });

            // Tri 2
            vertices.push(Vertex {
                x: v1.x,
                y: v1.y,
                z: v1.z,
                r: v1.r,
                g: v1.g,
                b: v1.b,
                section_id: mesh_id as u16,
            });
            vertices.push(Vertex {
                x: v3.x,
                y: v3.y,
                z: v3.z,
                r: v3.r,
                g: v3.g,
                b: v3.b,
                section_id: mesh_id as u16,
            });
            vertices.push(Vertex {
                x: v2.x,
                y: v2.y,
                z: v2.z,
                r: v2.r,
                g: v2.g,
                b: v2.b,
                section_id: mesh_id as u16,
            });
        }
    }

    Some((mesh_descs, vertices))
}
