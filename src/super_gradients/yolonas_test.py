
import sys
print(sys.path)
sys.path.append("/home/samuel/repos/super-gradients/src")


from super_gradients.training import models
from super_gradients.common.object_names import Models
import matplotlib 

matplotlib.style.use('ggplot')
matplotlib.use('tkagg')

net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
prediction = net.predict("https://www.aljazeera.com/wp-content/uploads/2022/12/2022-12-03T205130Z_851430040_UP1EIC31LXSAZ_RTRMADP_3_SOCCER-WORLDCUP-ARG-AUS-REPORT.jpg?w=770&resize=770%2C436&quality=80")
prediction.show()
print("elo")