from torch.utils.tensorboard import SummaryWriter
from utils.model_tool import get_model
from utils.DataLoader import get_dataLoader
import cv2

# write = SummaryWriter(r'./log')
# net, _ = get_model("S")
# data = get_dataLoader('./AVE_Dataset/Picture',"train","S",1,300,)
# data = iter(data)
# write.add_graph(net,data)
# write.close()

img1 = cv2.imread("S/test_result/-MtoePIUMdk/00.jpg")
print(img1.shape)
