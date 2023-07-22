#![allow(dead_code)]
use std::{fs::File, io::{Seek, SeekFrom, Write}, mem::size_of, ffi::c_void, f32::consts::PI, path::Path};
use byteorder::{ReadBytesExt, LittleEndian};
use gl::types::{GLenum, GLvoid};
use glam::Mat4;
use memoffset::offset_of;
use gl::types::GLfloat;
use glam::Vec3;

struct HeaderFMSH {
    file_magic: u32,
    n_submeshes: u32,
    offset_mesh_desc: u32,
    offset_vertex_data: u32
}

struct MeshDesc {
    vertex_start: u16,  // First vertex index for this model
    n_triangles: u16,   // Number of vertices for this model
    n_quads: u16,       // Number of vertices for this model
    x_min: i16,         // Axis aligned bounding box minimum X
    x_max: i16,         // Axis aligned bounding box maximum X
    y_min: i16,         // Axis aligned bounding box minimum Y
    y_max: i16,         // Axis aligned bounding box maximum Y
    z_min: i16,         // Axis aligned bounding box minimum Z
    z_max: i16,         // Axis aligned bounding box maximum Z
    pad: i16,
}

struct VertexPSX {
    x: i16,             // Position X
    y: i16,             // Position Y
    z: i16,             // Position Z
    r: u8,              // Color R
    g: u8,              // Color G
    b: u8,              // Color B
    u: u8,              // Texture Coordinate U
    v: u8,              // Texture Coordinate V
    extra: u8,          // In the first vertex, this is an index into the texture collection, which determines which texture to use. In the second vertex, this is the size of the triangle.
}

#[derive(Debug)]
struct Vertex {
    x: i16,
    y: i16,
    z: i16,
    section_id: u16,
}

struct VertexCol {
    x: i16,
    y: i16,
    z: i16,
    terrain_id: u16,
}

impl HeaderFMSH {
    fn read(buf_reader: &mut File) -> HeaderFMSH {
        HeaderFMSH {
            file_magic: buf_reader.read_u32::<LittleEndian>().unwrap(),
            n_submeshes: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_mesh_desc: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_vertex_data: buf_reader.read_u32::<LittleEndian>().unwrap(),
        }
    }
}

impl MeshDesc {
    fn read(buf_reader: &mut File) -> MeshDesc {
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
    fn read(buf_reader: &mut File) -> VertexPSX {
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
    fn read(buf_reader: &mut File) -> VertexCol {
        VertexCol {
            x: buf_reader.read_i16::<LittleEndian>().unwrap(),
            y: buf_reader.read_i16::<LittleEndian>().unwrap(),
            z: buf_reader.read_i16::<LittleEndian>().unwrap(),
            terrain_id: buf_reader.read_u16::<LittleEndian>().unwrap(),
        }
    }
}

const RESOLUTION: u32 = 128;
fn main() {
    // Get command line arguments and check if we have one
    let args: Vec<String> = std::env::args().collect();
    if (args.len() != 4) 
    || (!args[1].ends_with(".msh")) 
    || (!args[2].ends_with(".col")) 
    || (!args[3].ends_with(".vis")) 
    {
        println!("Usage: psx_vislist_generator.exe <input.msh> <input.col> <input.vis>");
        return;
    }

    // Open the collision model to use for finding the sampling positions
    let mut input_col = match File::open(Path::new(args[2].as_str())) {
        Ok(input_col) => input_col,
        Err(err) => {
            eprintln!("Failed to open file: {err}");
            return;
        }
    };

    // Open the visual mesh file to use for determining what is visible from the sampling positions
    let mut input_msh = match File::open(Path::new(args[1].as_str())) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to open file: {err}");
            return;
        }
    };

    // Open output file that will store the visibility lists
    let mut output_vis = match File::create(Path::new(args[3].as_str())) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to open file: {err}");
            return;
        }
    };

    // Read header
    let header_fmsh = HeaderFMSH::read(&mut input_msh);

    // Check file magic
    if header_fmsh.file_magic != 0x48534D46 {
        eprintln!("File identifier invalid! File is either not an FMSH or corrupted");
        return;
    }

    // Get all the mesh descriptions
    let mut mesh_descs = Vec::new();
    for i in 0..header_fmsh.n_submeshes {
        input_msh.seek(SeekFrom::Start(size_of::<HeaderFMSH>() as u64 + header_fmsh.offset_mesh_desc as u64 + (size_of::<MeshDesc>() * i as usize) as u64)).unwrap();
        mesh_descs.push(MeshDesc::read(&mut input_msh));
    }

    // Read all the vertex buffers
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut render_positions: Vec<Vertex> = Vec::new();
    let eye_height = 200;
    for (mesh_id, mesh) in mesh_descs.iter().enumerate() {
        // Seek to vertex data
        input_msh.seek(SeekFrom::Start(size_of::<HeaderFMSH>() as u64 + header_fmsh.offset_vertex_data as u64 + (mesh.vertex_start as u64 * size_of::<VertexPSX>() as u64))).unwrap();

        // Read all the triangles
        for _ in 0..mesh.n_triangles {
            let v0 = VertexPSX::read(&mut input_msh);
            let v1 = VertexPSX::read(&mut input_msh);
            let v2 = VertexPSX::read(&mut input_msh);
            vertices.push(Vertex { x: v0.x, y: v0.y, z: v0.z, section_id: mesh_id as u16 });
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id as u16 });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id as u16 });
            
        }

        // Read all the quads and convert them to triangles
        for _ in 0..mesh.n_quads {
            let v0 = VertexPSX::read(&mut input_msh);
            let v1 = VertexPSX::read(&mut input_msh);
            let v2 = VertexPSX::read(&mut input_msh);
            let v3 = VertexPSX::read(&mut input_msh);

            // Tri 1
            vertices.push(Vertex { x: v0.x, y: v0.y, z: v0.z, section_id: mesh_id as u16 });
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id as u16 });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id as u16 });

            // Tri 2
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id as u16 });
            vertices.push(Vertex { x: v3.x, y: v3.y, z: v3.z, section_id: mesh_id as u16 });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id as u16 });
        }
    }

    // Make sure it is a valid file
    let file_magic = input_col.read_u32::<LittleEndian>().unwrap();
    if file_magic != 0x4C4F4346 {
        eprintln!("Failed to open collision model!");
        return;
    }

    // Loop over all triangles
    let n_tris = input_col.read_u32::<LittleEndian>().unwrap() / 3;
    for _ in 0..n_tris {
        // Read vertex
        let v0 = VertexCol::read(&mut input_col);
        let v1 = VertexCol::read(&mut input_col);
        let v2 = VertexCol::read(&mut input_col);

        // Find the section it is in
        let mut section_id = 0;
        'outer_loop: for vertex in [&v0, &v1 ,&v2] {
            for mesh in &mesh_descs {
                if vertex.x >= mesh.x_min &&
                   vertex.x <= mesh.x_max &&
                   vertex.y >= mesh.y_min &&
                   vertex.y <= mesh.y_max &&
                   vertex.z >= mesh.z_min &&
                   vertex.z <= mesh.z_max {
                    break 'outer_loop;
                }
                section_id += 1;
            }
        }
        if section_id > mesh_descs.len() {
            continue;
        }

        // Generate normal
        let vtx0 = glam::vec3(v0.x as f32, v0.y as f32, v0.z as f32);
        let vtx1 = glam::vec3(v1.x as f32, v1.y as f32, v1.z as f32);
        let vtx2 = glam::vec3(v2.x as f32, v2.y as f32, v2.z as f32);
        let v01 = vtx1 - vtx0;
        let v02 = vtx2 - vtx0;
        let normal = v01.cross(v02).normalize();

        // If the geometry points upward enough, assume it's a spot the player can walk, and we should sample it
        if normal.y > 0.5 {
            render_positions.push(Vertex {
                x: (((v0.x as f32) + (v1.x as f32) + (v2.x as f32)) / 3.0) as i16,
                y: (((v0.y as f32) + (v1.y as f32) + (v2.y as f32)) / 3.0) as i16 - eye_height,
                z: (((v0.z as f32) + (v1.z as f32) + (v2.z as f32)) / 3.0) as i16,
                section_id: section_id as u16,
            })
        }
    }

    // Set up a basic OpenGL setup
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    // Create an invisible window
    let (window, _events) = glfw
        .create_window(RESOLUTION, RESOLUTION, "title", glfw::WindowMode::Windowed)
        .expect("Failed to create window.");
    glfw.make_context_current(Some(&window));

    // Init OpenGL
    gl::load_with(|f_name| glfw.get_proc_address_raw(f_name));
    unsafe {
        let error = gl::GetError();
        if error != gl::NO_ERROR {
            return;
        }
    }

    // Create shader
    let program;
    unsafe {
        program = gl::CreateProgram();
        load_shader_part(
            gl::VERTEX_SHADER,
            String::from("
                #version 460

                // Vertex input
                layout (location = 0) in vec3 i_position;
                layout (location = 1) in float i_section_id;

                // View matrix
                layout (location = 0) uniform mat4 u_matrix;
                layout (location = 1) uniform vec3 u_position;

                // Vert output
                out float o_section_id;
                out vec3 o_position;

                void main() {
                    gl_Position = u_matrix * vec4(i_position, 1);
                    o_position = i_position;
                    o_section_id = i_section_id;
                }
            "),
            program,
        );
        load_shader_part(
            gl::FRAGMENT_SHADER,
            String::from("
                #version 460

                in float o_section_id;
                in vec3 o_position;
                out vec4 frag_color;

                void main() {
                    frag_color = vec4(o_section_id / 255.0, gl_FragCoord.w / 4.0, 0.0, 1.0);
                    frag_color = vec4(
                        o_section_id / 255.0, 
                        gl_FragCoord.z, 
                        abs(mod(o_position.y, 512.0) - 256), 
                        256.0) / vec4(256.0);
                    frag_color.x = o_section_id / 255.0;
                }
            "),
            program,
        );

        gl::LinkProgram(program);
        gl::UseProgram(program);
    }

    // Upload the mesh to the GPU
    let mut vao = 0;
    let mut vbo = 0;
    unsafe {
        // Generate buffers
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

        // Define vertex layout
        gl::VertexAttribPointer(0, 3, gl::SHORT, gl::FALSE, size_of::<Vertex>() as i32, offset_of!(Vertex, x) as *const _);
        gl::VertexAttribPointer(1, 1, gl::UNSIGNED_SHORT, gl::FALSE, size_of::<Vertex>() as i32, offset_of!(Vertex, section_id) as *const _);
        gl::EnableVertexAttribArray(0);
        gl::EnableVertexAttribArray(1);

        // Upload the buffer
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (size_of::<Vertex>() * vertices.len()) as isize,
            &vertices[0] as *const Vertex as *const c_void,
            gl::STATIC_DRAW,
        );
    }

    // Set uniform variables
    let vectors = [
        glam::vec3(1.0, 0.0, 0.0),
        glam::vec3(-1.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
        glam::vec3(0.0, -1.0, 0.0),
        glam::vec3(0.0, 0.0, 1.0),
        glam::vec3(0.0, 0.0, -1.0),
    ];

    unsafe {
        let proj_matrix = Mat4::perspective_lh(PI / 4.0, 1.0, 0.01, 1000000.0);
        
        let mut view_matrix;
        

        let mut position = Vec3 {
            x: -11689.984,
            z: -8182.0672,
            y: -10932.224,
        };
        let mut vector_index = 0;
        let mut position_index = 0;    
        let buffer_size = (RESOLUTION * RESOLUTION * 4) as usize;
        let mut section_vislists = vec![0u128; header_fmsh.n_submeshes as usize];
        let mut buffer = vec![0u8; buffer_size];
        loop {
            // Set up render position
            position.x = render_positions[position_index].x as f32;
            position.y = render_positions[position_index].y as f32;
            position.z = render_positions[position_index].z as f32;
            
            // Set up view matrix
            view_matrix = Mat4::look_to_lh(
                position,
                vectors[vector_index],
                glam::vec3(0.0, -1.0, 0.0),
            );

            // Combined matrix
            let comb_mat = proj_matrix * view_matrix;

            // Render side of cubemap
            gl::UseProgram(program);
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::UniformMatrix4fv(0, 1, gl::FALSE, comb_mat.as_ref().as_ptr() as *const GLfloat);
            gl::Uniform3fv(1, 1, position.as_ref().as_ptr() as *const GLfloat);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::DrawArrays(gl::TRIANGLES, 0, vertices.len() as i32);

            gl::ReadPixels(0, 0, RESOLUTION as i32, RESOLUTION as i32, gl::RGBA, gl::UNSIGNED_BYTE, buffer.as_mut_ptr() as *mut GLvoid);
            for i in (0..buffer_size).step_by(4) {
                if buffer[i] != 255 {
                    section_vislists[render_positions[position_index].section_id as usize] |= 1 << buffer[i];
                }
            }

            // Show to screen
            //window.swap_buffers();
            //glfw.poll_events();

            // Close if we press X
            if window.should_close() {
                break;
            }

            // Go to next angles and positions
            vector_index += 1;
            if vector_index >= 6 {
                vector_index -= 6;
                position_index += 1;
                if position_index >= render_positions.len() {
                    println!("done!");
                    break;
                }
            }
        }

        // Write "FVIS" followed by number of sections
        output_vis.write_all(&0x53495646u32.to_le_bytes()).unwrap();
        output_vis.write_all(&(section_vislists.len() as u32).to_le_bytes()).unwrap();
        // Write nu

        for section in section_vislists {
            output_vis.write_all(&section.to_le_bytes()).unwrap();
        }
    }
}

fn load_shader_part(shader_type: GLenum, source: String, program: u32) {
    let source_len = source.len() as i32;

    unsafe {
        // Create shader part
        let shader = gl::CreateShader(shader_type);
        gl::ShaderSource(shader, 1, &source.as_bytes().as_ptr().cast(), &source_len);
        gl::CompileShader(shader);

        // Check for errors
        let mut result = 0;
        let mut log_length = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut result);
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut log_length);
        let mut error_message: Vec<u8> = vec![0; log_length as usize];
        gl::GetShaderInfoLog(
            shader,
            log_length,
            std::ptr::null_mut(),
            error_message.as_mut_ptr().cast(),
        );

        // Did we get an error?
        if log_length > 0 {
            println!(
                "Shader compilation error!\n{}",
                std::str::from_utf8(error_message.as_slice()).unwrap()
            )
        }

        // Attach to program
        gl::AttachShader(program, shader);
    }
}