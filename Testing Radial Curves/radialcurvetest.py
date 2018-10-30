from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *

import numpy as np

import glfw
import controls
import objloader
from csgl import *

import matplotlib.pyplot as plt

import time

import radialcurve

# Globals
window = None
null = c_void_p(0)

def init_opengl():
    global window
    # Initialize the library
    if not glfw.init():
        print("Failed to initialize GLFW\n",file=sys.stderr)
        return False

    # Open Window and create its OpenGL context
    window = glfw.create_window(1024, 768, "Radial Curve Test", None, None) 
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if not window:
        print("Failed to open GLFW window.\n",file=sys.stderr)
        glfw.terminate()
        return False

    # Initialize GLEW
    glfw.make_context_current(window)
    
    return True




#return GLuint
def LoadShaders(vertex_file_path,fragment_file_path):
    #Common OpenGL loadshader function
    
	# Create the shaders
	VertexShaderID = glCreateShader(GL_VERTEX_SHADER)
	FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER)

	# Read the Vertex Shader code from the file
	VertexShaderCode = ""
	with open(vertex_file_path,'r') as fr:
		for line in fr:
			VertexShaderCode += line

	FragmentShaderCode = ""
	with open(fragment_file_path,'r') as fr:
		for line in fr:
			FragmentShaderCode += line

	# Compile Vertex Shader
	glShaderSource(VertexShaderID, VertexShaderCode)
	glCompileShader(VertexShaderID)

	# Check Vertex Shader
	result = glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS)
	if not result:
		raise RuntimeError(glGetShaderInfoLog(VertexShaderID))

	# Compile Fragment Shader
	glShaderSource(FragmentShaderID,FragmentShaderCode)
	glCompileShader(FragmentShaderID)

	# Check Fragment Shader
	result = glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS)
	if not result:
		raise RuntimeError(glGetShaderInfoLog(FragmentShaderID))



	# Link the program
	ProgramID = glCreateProgram()
	glAttachShader(ProgramID, VertexShaderID)
	glAttachShader(ProgramID, FragmentShaderID)
	glLinkProgram(ProgramID)

	# Check the program
	result = glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS)
	if not result:
		raise RuntimeError(glGetShaderInfoLog(ProgramID))

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
    

def key_event(window,key,scancode,action,mods):
    if action == glfw.PRESS and key == glfw.KEY_F:
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    if action == glfw.PRESS and key == glfw.KEY_G:
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );


def main():
    time_start = time.time()
    
    if not init_opengl():
        return
    
    #key events
    glfw.set_input_mode(window,glfw.STICKY_KEYS,GL_TRUE) 
    glfw.set_cursor_pos(window, 1024/2, 768/2)
    glfw.set_key_callback(window,key_event)
    
    glClearColor(0,0,0.4,0)
    
    # Enable depth test
    glEnable(GL_DEPTH_TEST);
    # Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS); 
    
    #Faces wont display if normal facing camera ie inside of mesh
    glEnable(GL_CULL_FACE)
    
    #Disable vsync
    glfw.swap_interval(0)
    
    

    shader_program = LoadShaders("VertexShader.vertexshader", "FragmentShader.fragmentshader")
    
    vao1 = glGenVertexArrays(1)
    glBindVertexArray(vao1)

    #Read obj
    vertices,faces,uvs,normals,colors = objloader.load("0000.obj")
    vertex_data,uv_data,normal_data = objloader.process_obj( vertices,faces,uvs,normals,colors)
        
    
    #convert to ctype
    #vertex_data = objloader.generate_2d_ctypes(vertex_data)
    #normal_data = objloader.generate_2d_ctypes(normal_data)
    #uv_data = objloader.generate_2d_ctypes(uv_data)
    
    vertex_data = np.array(vertex_data).astype(np.float32)
    normal_data = np.array(normal_data).astype(np.float32)
    uv_data = np.array(uv_data).astype(np.float32)
    
    

    # Load OBJ in to a VBO
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4 * 3, vertex_data, GL_STATIC_DRAW)
    #glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4 * 3, vert_data, GL_STATIC_DRAW)
    
    normal_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, normal_buffer)
    glBufferData(GL_ARRAY_BUFFER, len(normal_data) * 4 * 3, normal_data, GL_STATIC_DRAW)
    
    uv_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, uv_buffer)
    glBufferData(GL_ARRAY_BUFFER, len(uv_data) * 4 * 2, uv_data, GL_STATIC_DRAW)
    
    #Unbind
    glBindVertexArray(0)
    
    
    #Radial VAO
    vao2 = glGenVertexArrays(1)
    
    #Radial curves data
    #radialcurve.create_vertex_face_dict(vertices, faces)
    time_end = time.time()
    print("Time passed until radial functions: ", time_end-time_start, " seconds")
    time_start = time.time()
    radialcurve.vertex_face_sort(vertices, faces)
    time_end = time.time()
    print("Array Time: ", time_end-time_start)
    segment = 20
    time_start = time.time()
    radial_data_unfinished = radialcurve.get_radial_curves(segment,
                                                           (5.163908, -21.303692), (0.601237, 58.022675),
                                                           (-60, 0), (60, 0),
                                                           (0, 60), (0, -60),
                                                           (-40, -20), (20, -40)
                                                            )
    time_end = time.time()
    print("Radial Curve Time: ", time_end-time_start)
    
    radial_data = radial_data_unfinished[3]
    #Increase z by 1 so easier to see
    for point in radial_data:
        point[2] = point[2] + 1
        #point[2] = 25
        
    
    #matplot
    #`x = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    x = [i/(segment+1) for i in range(segment+2)]
    y = []
    for lines in radial_data_unfinished:
        for data in lines:
            y.append(data[2])
        plt.plot(x, y)
        y = []
    #print(y)
        

    plt.xlabel("Distance")
    plt.ylabel("Depth")
    
    plt.show()
    
    #print(radial_data)
    #radial_data = [[5.163908, -21.303692, 25], [0.601237, 58.022675, 25]]
    radial_data = objloader.generate_2d_ctypes(radial_data)
    
    #Radial Curves
    radial_buffer = glGenBuffers(1)
    array_type = GLfloat * len(radial_data)
    glBindBuffer(GL_ARRAY_BUFFER, radial_buffer)
    #glBufferData(GL_ARRAY_BUFFER, len(radial_data) * 4, array_type(*radial_data), GL_STATIC_DRAW)
    rad_data = np.array(radial_data).astype(np.float32)
    glBufferData(GL_ARRAY_BUFFER, len(radial_data) * 4 * 3, rad_data, GL_STATIC_DRAW)
    #Unbind
    glBindVertexArray(0)
    
    #####
    
    #Handles
    light_id = glGetUniformLocation(shader_program, "LightPosition_worldspace");
    matrix_id = glGetUniformLocation(shader_program, "MVP")
    view_matrix_id = glGetUniformLocation(shader_program, "V")
    model_matrix_id = glGetUniformLocation(shader_program, "M")
    
    
    
    # Enable key events
    glfw.set_input_mode(window,glfw.STICKY_KEYS,GL_TRUE) 
    
    #Line Width
    glEnable(GL_LINE_SMOOTH)
    glLineWidth(	3)
    
    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)
        
        controls.computeMatricesFromInputs(window)
        pm = controls.getProjectionMatrix()
        vm = controls.getViewMatrix()
        #at origin
        mm = mat4.identity()
        mvp = pm * vm * mm
        
        
        glUniformMatrix4fv(matrix_id, 1, GL_FALSE,mvp.data)
        glUniformMatrix4fv(model_matrix_id, 1, GL_FALSE, mm.data);
        glUniformMatrix4fv(view_matrix_id, 1, GL_FALSE, vm.data);

        lightPos = vec3(0,0,50)
        glUniform3f(light_id, lightPos.x, lightPos.y, lightPos.z)
        
        
        ##VAO1################
        glBindVertexArray(vao1)
        
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glVertexAttribPointer(
			0,                  # attribute
			3,                  # len(vertex_data)
			GL_FLOAT,           # type
			GL_FALSE,           # ormalized?
			0,                  # stride
			null           		# array buffer offset (c_type == void*)
			)
        
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, normal_buffer);
        glVertexAttribPointer(
            1,                                  # attribute
            3,                                  # size
            GL_FLOAT,                           # type
            GL_FALSE,                           # ormalized?
            0,                                  # stride
            null                                # array buffer offset (c_type == void*)
        )
        
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, uv_buffer);
        glVertexAttribPointer(
            2,                                  # attribute
            2,                                  # size
            GL_FLOAT,                           # type
            GL_FALSE,                           # ormalized?
            0,                                  # stride
            null                                # array buffer offset (c_type == void*)
        )
        

		# Draw
        glDrawArrays(GL_TRIANGLES, 0, len(vertex_data)) 

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        
        glBindVertexArray(0)
        
        ######################################
        
        #VAO2
        glBindVertexArray(vao2)
        
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, radial_buffer);
        glVertexAttribPointer(
            0,                                  # attribute
            3,                                  # size
            GL_FLOAT,                           # type
            GL_FALSE,                           # ormalized?
            0,                                  # stride
            null                                # array buffer offset (c_type == void*)
        )
        
        
        glDrawArrays(GL_LINE_STRIP, 0, len(radial_data))
        #glDrawArrays(GL_TRIANGLES, 0, len(radial_data))

        glBindVertexArray(0)
        
        
        glfw.swap_buffers(window)
        
        glfw.poll_events()
    
    glBindVertexArray(vao1)
    glDeleteBuffers(1, [vertex_buffer])
    glDeleteBuffers(1, [normal_buffer])
    glBindVertexArray(0)
    glDeleteVertexArrays(1, [vao1])
    
    glBindVertexArray(vao2)
    glDeleteBuffers(1, [radial_buffer])
    glDeleteVertexArrays(1, [vao2])
    
    glDeleteProgram(shader_program)
    
    glfw.terminate();
    
    
if __name__ == "__main__":
    main()
