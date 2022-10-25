# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:11:40 2022

@author: dguti
"""

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# modelo, etiqeutas e imagen de ejemplo
model_file = "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
label_file = "inat_bird_labels.txt"
image_file = "parrot.jpg"

#%%

# Inicializar el interprete de tensorflow lite
interpreter = edgetpu.make_interpreter(model_file)
# reserva de los tensores
interpreter.allocate_tensors()

# Ajustamos datos entrada
size = common.input_size(interpreter) # datos entrada
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Ejecutar inferencia
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

#%% Comprobamos el resultados

labels = dataset.read_label_file(label_file)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))