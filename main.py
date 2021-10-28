from homepage2vec.model import WebsiteClassifier

model_path = "/Users/piccardi/repos/multilang-web-embedding/training/results/final_1000_100_posw_heldout"

model = WebsiteClassifier(model_path=model_path, cpu_threads_count=1, dataloader_workers=1)

webpages = model.embed_and_predict(['www.repubblica.it'])

print(webpages[0].scores)