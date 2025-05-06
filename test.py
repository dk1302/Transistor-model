from cnn import *
import cnn

if __name__ == "__main__":
    model = cnn.use_model('datasets/val.csv', index=2)
    model.run('models/shape_v1.pth', 'models/scale_v1.pth', v=1)
    model.run('models/shape_v5.pth', 'models/scale_v5.pth', v=5)
    model.run('models/shape_v10.pth', 'models/scale_v10.pth', v=10)