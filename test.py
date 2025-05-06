from cnn import *
import cnn

if __name__ == "__main__":
    model = cnn.use_model('val.csv', index=10)
    model.run('models/cnn_model_v1.pth', 'models/scale_v1.pth', v=1)
    model.run('models/cnn_model_v5.pth', 'models/scale_v5.pth', v=5)
    model.run('models/cnn_model_v10.pth', 'models/scale_v10.pth', v=10)