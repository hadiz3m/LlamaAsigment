# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(0, 128, 0.1)
# y = np.sin(x)


# plt.plot(x, y)

# fig, ax = plt.subplots()
# ax.plot(x, y)

# plt.show()

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')
from transformers import AutoModel, AutoTokenizer
from torchview import draw_graph
model1 = AutoModel.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")

# model_graph = draw_graph(model1, input_data=inputs)

# model_graph.visual_graph


# from transformers import AutoModel
# model1 = AutoModel.from_pretrained("bert-base-uncased")

from torchviz import make_dot, make_dot_from_trace
make_dot(model1)