import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points = {
    'blue': [[2, 4, 1], [1, 3, 2], [2, 3, 1], [3, 2, 2], [1, 2, 1]],
    'red':  [[5, 6, 2], [4, 5, 3], [4, 6, 2], [6, 6, 3], [5, 4, 2]]
}

new_point = [4, 5, 2]


def euclidian_distance(p,q):
    return np.sqrt(np.sum((np.array(p)- np.array(q))**2)) #distance formula
                   
class KNN:
    def __init__(self, k=3):
        self.k=k
        self.point=None 
        
    def fit(self, points):
        self.points=points #to fit the model(class object) the defined points
    
    def predict(self, new_point):
        distances=[]
        
        for category in self.points: #for the loop of categories
            for point in self.points[category]: #for each point in each category (all the points in blue/red)
                distance=euclidian_distance(point, new_point) #euclidian distance formula
                distances.append([distance, category]) #appending both the distance from the new point and category of the class the point is from
        
        categories=[category[1] for category in sorted(distances)[:self.k]]  #will return K categories based on smallest distances to the new point
        result=Counter(categories).most_common(1)[0][0] #gets the most common category out of 3 categories like blue blue red for example
        return result
    
cls=KNN() #constructor
cls.fit(points) #points assign
print(cls.predict(new_point)) 


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], color="#104DCA", s=60)

for point in points['red']:
    ax.scatter(point[0], point[1], point[2], color="#FF0000", s=60)

new_class = cls.predict(new_point)
new_color = "#FF0000" if new_class == "red" else "#104DCA"

ax.scatter(new_point[0], new_point[1], new_point[2], color=new_color, s=200)

for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]],color='#104DCA', linestyle="--")
    
for point in points['red']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]],color='#FF0000', linestyle="--")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
        