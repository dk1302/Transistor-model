from cnn import *
import cnn

if __name__ == "__main__":
    model = cnn.use_model('val.csv', index=0)
    model.run('cnn_model_v1.pth', v=1)
    model.run('cnn_model_v5.pth', v=5)
    model.run('cnn_model_v10.pth', v=10)