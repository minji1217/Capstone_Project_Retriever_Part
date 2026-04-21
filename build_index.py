import json 
import faiss
import time
import numpy as np
import pickle
from embedder import SpecterEmbedder
import config
import utils


# 1. 원본 논문 DB(JSON) 불러오기
def load_papers(file_path): 
    print(f"[{file_path}] 원본 데이터 로딩 중...")
    with open(file_path, "r", encoding = "utf-8"):
            


def build_offline_system():
    return 