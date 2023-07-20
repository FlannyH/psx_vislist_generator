use std::{fs::File, io::{Seek, SeekFrom}, mem::size_of, ffi::c_void, f32::consts::PI, collections::HashMap};
use byteorder::{ReadBytesExt, LittleEndian};
use gl::types::{GLenum, GLint, GLsizei, GLvoid};
use glam::Mat4;
use memoffset::offset_of;
use gl::types::GLfloat;
use glam::Vec3;
use glfw::Context;

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

struct Vertex {
    x: i16,
    y: i16,
    z: i16,
    section_id: u16,
}

impl HeaderFMSH {
    fn read(buf_reader: &mut File) -> HeaderFMSH {
        return HeaderFMSH {
            file_magic: buf_reader.read_u32::<LittleEndian>().unwrap(),
            n_submeshes: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_mesh_desc: buf_reader.read_u32::<LittleEndian>().unwrap(),
            offset_vertex_data: buf_reader.read_u32::<LittleEndian>().unwrap(),
        }
    }
}

impl MeshDesc {
    fn read(buf_reader: &mut File) -> MeshDesc {
        return MeshDesc {
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
        return VertexPSX {
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

fn main() {
    // Open file
    let mut file = match File::open("D:/Projects/Git/ShooterPSX/assets/models/level.msh") {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to open file: {err}");
            return;
        }
    };

    // Read header
    let header_fmsh = HeaderFMSH::read(&mut file);

    // Check file magic
    if header_fmsh.file_magic != 0x48534D46 {
        eprintln!("File identifier invalid! File is either not an FMSH or corrupted");
        return;
    }

    // Get all the mesh descriptions
    let mut mesh_descs = Vec::new();
    for i in 0..header_fmsh.n_submeshes {
        file.seek(SeekFrom::Start(size_of::<HeaderFMSH>() as u64 + header_fmsh.offset_mesh_desc as u64 + (size_of::<MeshDesc>() * i as usize) as u64)).unwrap();
        mesh_descs.push(MeshDesc::read(&mut file));
    }

    // Read all the vertex buffers
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut render_positions: Vec<Vertex> = Vec::new();
    let eye_height = 200;
    let mut mesh_id = 0;
    for mesh in mesh_descs {
        // Seek to vertex data
        file.seek(SeekFrom::Start(size_of::<HeaderFMSH>() as u64 + header_fmsh.offset_vertex_data as u64 + (mesh.vertex_start as u64 * size_of::<VertexPSX>() as u64))).unwrap();

        // Read all the triangles
        for i in 0..mesh.n_triangles {
            println!("mesh_id {mesh_id}, tri {i}, vertex {}", vertices.len());
            let v0 = VertexPSX::read(&mut file);
            let v1 = VertexPSX::read(&mut file);
            let v2 = VertexPSX::read(&mut file);
            let vtx0 = glam::vec3(v0.x as f32, v0.y as f32, v0.z as f32);
            let vtx1 = glam::vec3(v1.x as f32, v1.y as f32, v1.z as f32);
            let vtx2 = glam::vec3(v2.x as f32, v2.y as f32, v2.z as f32);
            let v01 = vtx1 - vtx0;
            let v02 = vtx2 - vtx0;
            let normal = v01.cross(v02).normalize();
            if normal.y > 0.5 {
                render_positions.push(Vertex {
                    x: (((v0.x as f32) + (v1.x as f32) + (v2.x as f32)) / 3.0) as i16,
                    y: (((v0.y as f32) + (v1.y as f32) + (v2.y as f32)) / 3.0) as i16 - eye_height,
                    z: (((v0.z as f32) + (v1.z as f32) + (v2.z as f32)) / 3.0) as i16,
                    section_id: mesh_id,
                })
            }
            vertices.push(Vertex { x: v0.x, y: v0.y, z: v0.z, section_id: mesh_id });
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id });
            
        }

        // Read all the quads and convert them to triangles
        for i in 0..mesh.n_quads {
            println!("mesh_id {mesh_id}, quad {i}, vertex {}", vertices.len());
            let v0 = VertexPSX::read(&mut file);
            let v1 = VertexPSX::read(&mut file);
            let v2 = VertexPSX::read(&mut file);
            let v3 = VertexPSX::read(&mut file);

            let vtx0 = glam::vec3(v0.x as f32, v0.y as f32, v0.z as f32);
            let vtx1 = glam::vec3(v1.x as f32, v1.y as f32, v1.z as f32);
            let vtx2 = glam::vec3(v2.x as f32, v2.y as f32, v2.z as f32);
            let v01 = vtx1 - vtx0;
            let v02 = vtx2 - vtx0;
            let normal = v01.cross(v02).normalize();
            if normal.y > 0.5 {
                render_positions.push(Vertex {
                    x: (((v0.x as f32) + (v1.x as f32) + (v2.x as f32)) / 3.0) as i16,
                    y: (((v0.y as f32) + (v1.y as f32) + (v2.y as f32)) / 3.0) as i16 - eye_height,
                    z: (((v0.z as f32) + (v1.z as f32) + (v2.z as f32)) / 3.0) as i16,
                    section_id: mesh_id,
                })
            }
            vertices.push(Vertex { x: v0.x, y: v0.y, z: v0.z, section_id: mesh_id });
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id });

            // Tri 1
            vertices.push(Vertex { x: v0.x, y: v0.y, z: v0.z, section_id: mesh_id });
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id });

            // Tri 2
            vertices.push(Vertex { x: v1.x, y: v1.y, z: v1.z, section_id: mesh_id });
            vertices.push(Vertex { x: v3.x, y: v3.y, z: v3.z, section_id: mesh_id });
            vertices.push(Vertex { x: v2.x, y: v2.y, z: v2.z, section_id: mesh_id });
        }

        mesh_id += 1;
    }

    // Set up a basic OpenGL setup
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    // Create an invisible window
    let (mut window, _events) = glfw
        .create_window(256, 256, "title", glfw::WindowMode::Windowed)
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

                void main() {
                    gl_Position = u_matrix * vec4(i_position * vec3(-0.000244140625) - u_position * vec3(-0.000244140625), 1);
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
                out vec4 frag_color;

                void main() {
                    frag_color = vec4(o_section_id / 256.0, 0.0, 0.0, 1.0);
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
        let proj_matrix = Mat4::perspective_rh(PI / 4.0, 1.0, 0.1, 10000.0);
        
        let mut view_matrix = Mat4::look_at_rh(
            glam::vec3(0.0, 0.0, 0.0),
            vectors[0],
            glam::vec3(0.0, 1.0, 0.0),
        );
        let mut position = Vec3 {
            x: -11689.984,
            z: -8182.0672,
            y: -10932.224,
        };
        let mut counter = 0;
        let mut vector_index = 0;
        let mut position_index = 0;    
        let buffer_size = (256 * 256 * 4) as usize;
        let mut section_vislists = HashMap::<u16, u128>::new();
        let mut buffer = vec![0u8; buffer_size];
        loop {
            while counter < 300 {
                window.swap_buffers();
                glfw.poll_events();
                counter += 1;
                continue;
            }
            // Set up view matrix
            view_matrix = Mat4::look_at_rh(
                glam::vec3(0.0, 0.0, 0.0),
                vectors[vector_index],
                glam::vec3(0.0, 1.0, 0.0),
            );

            // Set up render position
            position.x = render_positions[position_index].x as f32;
            position.y = render_positions[position_index].y as f32;
            position.z = render_positions[position_index].z as f32;

            // Render side of cubemap
            gl::UseProgram(program);
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::UniformMatrix4fv(0, 1, gl::FALSE, (proj_matrix * view_matrix).as_ref().as_ptr() as *const GLfloat);
            gl::Uniform3fv(1, 1, position.as_ref().as_ptr() as *const GLfloat);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::DrawArrays(gl::TRIANGLES, 0, vertices.len() as i32);

            // Show to screen
            window.swap_buffers();
            glfw.poll_events();

            gl::ReadPixels(0, 0, 256, 256, gl::RGBA, gl::UNSIGNED_BYTE, buffer.as_mut_ptr() as *mut GLvoid);
            for i in 0..buffer_size {
                if buffer[i] != 255 {
                    let curr_vislist = section_vislists.entry(render_positions[position_index].section_id).or_insert(0);
                    *curr_vislist |= 1 << buffer[i];
                }
            }

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
        for (key, value) in section_vislists {
            println!("{key}: {value:X}")
        }
    }

    return;
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