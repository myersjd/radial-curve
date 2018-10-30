import numpy as np
from scipy import spatial


vertex_face_array = None
kd_tree = None
kd_tree_data = None
front_depth = float("-inf")

radius = 0
    

def vertex_face_sort(vertices, faces):
    #Remove all z values for 2d array
    global kd_tree_data
    kd_tree_data = [(v[0], v[1]) for v in vertices ]
        
    #create 2d kd_tree
    global kd_tree
    kd_tree = spatial.KDTree(kd_tree_data)
    
    #create dictionary
    global vertex_face_array    
    vertex_face_array = np.empty(len(vertices), dtype=np.object_) #Empty numpy array
    for x in range(len(vertex_face_array)):
        vertex_face_array[x] = [] # Set empty list in cell
    

    
    #Assign closest depth during loop
    global front_depth
    
    #Calculate sum of all distanes and take the average
    sum_ = 0
    counter = 0
    for face in faces:        
        vertex1 = np.array(vertices[face[0]-1])
        vertex2 = np.array(vertices[face[1]-1])
        vertex3 = np.array(vertices[face[2]-1])
        
        if vertex1[2] > front_depth:
            front_depth = vertex1[2]+1
        if vertex2[2] > front_depth:
            front_depth = vertex2[2]+1
        if vertex3[2] > front_depth:
            front_depth = vertex3[2]+1
        
        triangle = [vertex1, vertex2, vertex3]
        
        #add to dictionary for each vertex
        vertex_face_array[face[0]-1].append(triangle)
        vertex_face_array[face[1]-1].append(triangle)
        vertex_face_array[face[2]-1].append(triangle)
        
        #Add distances and increase coutner
        sum_ = sum_ + np.linalg.norm(vertex1-vertex2) + np.linalg.norm(vertex1-vertex3) + np.linalg.norm(vertex2-vertex3)
        counter = counter + 3
        
    #Set radius to average distance
    global radius
    radius = sum_/counter
        
        
        
        
        
def ray_triangle_intersect(point, triangle):
    
    #Using Möller–Trumbore intersection algorithm
    #Check and return point of intersected tri
        
    default_return = [float("-inf"), float("-inf"), float("-inf")]
    
    #Point away from screen
    ray_vector = np.array([0, 0, -1])
    ray_origin = np.array(point)
        
    
    epsilon=0.0000001
    #1e-6
                
        
    vertex0 = np.array(triangle[0])
    vertex1 = np.array(triangle[1])
    vertex2 = np.array(triangle[2])
            
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    
    if(a > -epsilon and a < epsilon):
        return default_return
    
    f = 1/a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    
    if(u < 0.0 or u > 1):
        return default_return
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    
    if(v < 0.0 or u+v > 1.0):
        return default_return
 
    t = f * np.dot(edge2, q)
    if(t > epsilon): #Ray Intersection
        intersection = ray_origin + ray_vector*t
        return intersection
        
    #If arriving at this point, there is a line intersection but not a ray intersection
            

    return default_return



def point_in_triangle(A, B, C, P): 

    #barycentric variables
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    P = np.array(P)

    u = B - A
    v = C - A
    w = P - A

    vCrossW = np.cross(v, w)
    vCrossU = np.cross(v, u)
	
    #Test sign of r
    if (np.dot(vCrossW, vCrossU) < 0):
        return False

    uCrossW = np.cross(u, w);
    uCrossV = np.cross(u, v);

    #Test sign of t
    if (np.dot(uCrossW, uCrossV) < 0):
        return False

 

    #r and t both > 0.
    #as long as their sum is <= 1, each must be less <= 1
	
    denom = np.linalg.norm(uCrossV)

    r = np.linalg.norm(vCrossW) / denom

    t = np.linalg.norm(uCrossW) / denom

    return (r + t <= 1)


def get_radial_curves(segment, *points):

    #Find nearest vertex then use dictionary to match vertex to all posible faces
    #Then find intersection at each segment interval
    
    global vertex_face_dictionary
    global kd_tree
    global kd_tree_data
    global radius
    global step
    global front_depth
        
    lines = []
    
    
    for i in range(0, len(points), 2): #Loop through pairs of points
        
        
        
        #Handle when i is the final point and no matching index of i+1 exists
        if(i+1 >= len(points)):
            continue
        
        p1 = points[i]
        p2 = points[i+1]
        
        line = []
        for x in range(segment+2): #Loop through segments of line
            point = [
                    p1[0]+(x/(segment+1))*(p2[0]-p1[0]),
                    p1[1]+(x/(segment+1))*(p2[1]-p1[1]),
                    #p1[2]+(x/(segment+1))*(p2[2]-p1[2])
                    front_depth
                    ]

            
            
            #Use KDTree to find nearest neighbor
            #Get nearest neighbors indecies in radius
            nearest_neighbors_indecies = kd_tree.query_ball_point([point[0], point[1]], radius)
            #List of nearest neighbors in radius of point (in 2d terms)
            #nearest_neighbors = [kd_tree_data[i] for i in nearest_neighbors_indecies]
            
            #print("Found using radius ", search_radius, " from radius ", radius)
            
            #No neighbors found
            if not nearest_neighbors_indecies:
                print("No nearest neighbors found for query ball search!")
                point[2] = float("-inf")
                line.append(point)
                continue    
            
            #search for intersection
            intersected_neighbor_depth = float("-inf")
            for nn_index in nearest_neighbors_indecies:
                #Find intersection of nearest neighbor with face using dictionary
                possible_faces = vertex_face_array[nn_index] #returns list of faces

                for face in possible_faces: #check each face for ray triangle intersection
                    #Cast ray to tri
                    temp = ray_triangle_intersect(point, face)[2]
                    if intersected_neighbor_depth < temp: #If temp closer to screen
                        intersected_neighbor_depth = temp
                    
                    #Cast ray to tri by check if inside triangle -- Slower than casting ray to every face
                    """if point_in_triangle([face[0][0], face[0][1]], [face[1][0], face[1][1]], [face[2][0], face[2][1]], [point[0], point[1]]):
                        #If inside triangle, find ray-triangle intersection
                        temp = ray_triangle_intersect(point, face)[2]
                        if intersected_neighbor_depth < temp: #If temp closer to screen
                            intersected_neighbor_depth = temp"""



            #Set new depth
            point[2] = intersected_neighbor_depth
            
            #Add point to current line
            line.append(point)
                    
        #Add line to list of lines
        lines.append(line)
        
    return lines
    

def main():
    vertex = [1, 1, 1]
    face = [[0, -1, 0], [-1, 1, 0], [1, 1, 0]]
    print(point_in_tri(face[0][0], face[0][1], face[1][0], face[1][1], face[2][0], face[2][1], vertex[0], vertex[1]))
    intersection = ray_triangle_intersect(vertex, face)
    print(intersection)
    
if __name__ == "__main__":
        main()
    
    