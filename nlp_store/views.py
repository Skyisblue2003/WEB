from django.shortcuts import render
from .models import Product 
from django.conf import settings
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
from pythainlp.util import normalize
from pythainlp.spell import correct
import os

# โหลดและเตรียมข้อมูลจาก CSV
file_path = os.path.join(settings.BASE_DIR, 'nlp_store', 'static', 'fashion_products_thai.csv')
df = pd.read_csv(file_path)

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

def normalize_input(text):
    text = normalize(text)
    text = re.sub(r"\s+", "", text)
    return correct(text)

# เตรียมข้อมูลสำหรับการแนะนำ
df["cleaned_product"] = df["product"].apply(clean_text)
df["cleaned_description"] = df["description"].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["cleaned_description"])

sentences = [row.split() for row in df["cleaned_description"]]
ft_model = FastText(vector_size=100, window=5, min_count=1, workers=4)
ft_model.build_vocab(sentences)
ft_model.train(sentences, total_examples=len(sentences), epochs=10)

def get_most_similar_word(user_input):
    try:
        similar_words = ft_model.wv.most_similar(user_input, topn=5)
        return [word for word, _ in similar_words]
    except:
        return []

def recommend_product(product_name):
    product_name = clean_text(normalize_input(product_name))

    matching_products = df[df["cleaned_description"].str.contains(product_name, na=False) |
                           df["cleaned_product"].str.contains(product_name, na=False)]

    if not matching_products.empty:
        return ("", list(zip(matching_products["product"], matching_products["description"], matching_products["image_path"])))

    similar_words = get_most_similar_word(product_name)
    if similar_words:
        suggestion_text = f"ไม่พบสินค้านั้น\nคุณอาจหมายถึง: {', '.join(similar_words)}"
        similar_matches = df[df["cleaned_description"].apply(lambda x: any(word in x for word in similar_words)) |
                             df["cleaned_product"].apply(lambda x: any(word in x for word in similar_words))]
        return (suggestion_text, list(zip(similar_matches["product"], similar_matches["description"], similar_matches["image_path"][:5])))
    else:
        return ("ไม่พบสินค้านั้นและไม่มีคำที่ใกล้เคียง", [])

def home(request):
    query = request.GET.get('search', '')  # รับค่าจาก query string หรือ form
    products = Product.objects.filter(product__icontains=query)  # เปลี่ยน name -> product

    if query:
        suggestion_text, similar_products = recommend_product(query)
        return render(request, 'nlp_store/home.html', {
            'products': products, 
            'query': query,
            'suggestion_text': suggestion_text,
            'similar_products': similar_products
        })

    return render(request, 'nlp_store/home.html', {
        'products': products,
        'query': query,
        'suggestion_text': "",
        'similar_products': []
    })
