from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path

model_path = "liuhaotian/llava-v1.6-vicuna-13b"
#model_path = "liuhaotian/llava-v1.6-34b"
prompt1 = "Describe this image in detail."
#prompt2 = "Describe this sequence of images in detail."
#prompt3 = "What is happening in this image? Is there violence? Generate a reasoning for answering the question based on the image, and infer the answer based on the image, the question and the reasoning."
image_file = "/data/jaca.pereira/datasets/ucf_crime/Videos/Assault/Assault020_x264/0103.png"
#image_file = "image_for_inference/garbage_kitten_2.png"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt1,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "load_4bit": True
})()

eval_model(args)
