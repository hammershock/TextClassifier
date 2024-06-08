import os
import random
from typing import List, Iterator, Tuple

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from joblib import Memory


memory = Memory(location="./.cache", verbose=0)


def tokenize(document):
    tokens = list(jieba.cut(document))
    return tokens


def load_txt(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return "".join(f.readlines())


def load_txt_dir(txt_dir) -> Iterator[str]:
    for file in os.listdir(txt_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(txt_dir, file)
            yield load_txt(file_path)


@memory.cache
def load_data(positive_dir, negative_dir) -> Tuple[List[List[str]], List[int]]:
    positive_samples = [(text, 1) for text in load_txt_dir(positive_dir)]
    negative_samples = [(text, 0) for text in load_txt_dir(negative_dir)]
    data = positive_samples + negative_samples
    random.shuffle(data)
    texts, labels = zip(*data)
    tokens = [tokenize(text) for text in tqdm(texts, "tokenizing")]
    return tokens, labels


if __name__ == '__main__':
    tokens, labels = load_data("./ad_classify/positive", "./ad_classify/negative")
    documents = [' '.join(token_list) for token_list in tokens]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    y = np.array(labels)

    model = LogisticRegression()
    model.fit(X, y)

    # 获取词汇列表和对应的权重（系数）
    feature_names = vectorizer.get_feature_names_out()
    # print(feature_names.tolist())
    coefficients = model.coef_[0]

    # 将词汇和权重结合起来
    word_importance = sorted(zip(feature_names, coefficients), key=lambda x: x[1], reverse=True)

    # 打印前10个对分类结果贡献最大的词汇
    print("Top 10 words contributing to classification:")
    with open("./ad_words_2000.txt", "w") as f:
        for word, coef in word_importance[:2000]:
            # print(f"{word}: {coef}")
            f.write(f"{word}\t{coef}\n")

    # # 打印前10个对分类结果贡献最小（负贡献）的词汇
    # print("\nTop 10 words negatively contributing to classification:")
    # for word, coef in word_importance[-100:]:
    #     print(f"{word}: {coef}")
