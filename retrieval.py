import contextlib
import os
from pathlib import Path
import shutil
import warnings
from file_context import get_all_names
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from retrieval_data import fake_data
from main import hvm3_dataset
from ragatouille.RAGTrainer import RAGTrainer
from ragatouille.RAGPretrainedModel import RAGPretrainedModel

os.environ['COLBERT_LOAD_TORCH_EXTENSION_VERBOSE'] = "False"

app = FastAPI()


class SearchRequest(BaseModel):
  query: str
  k: int = 5


class SearchResult(BaseModel):
  index: int
  symbol_name: str
  score: float


class SearchResponse(BaseModel):
  results: List[SearchResult]


# Global Colbert instance
colbert1 = None

# @app.on_event("startup")
# async def startup_event():
#     global colbert
#     colbert = Colbert("default")
#     colbert.search("test")

# @app.post("/search", response_model=SearchResponse)
# async def search(request: SearchRequest):
#     results = colbert.search(request.query, k=request.k)
#     print(results)
#     search_results = [
#         SearchResult(
#             index=int(result["document_metadata"]["index"]),
#             symbol_name=result["document_metadata"]["symbol_name"],
#             score=float(result["score"])
#         )
#         for result in results
#     ]
#     return SearchResponse(results=search_results)


class Colbert:
  def __init__(self, model_name: str = "colbert-ir/colbertv2.0", index_name: str = "default"):
    self.model_name = model_name
    self.index_name = index_name

  def search(self, query, k=10):
    self.create_index_if_not_exists()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
      results = self.rag.search(query, k=k, index_name=self.index_name)
    return results

  def find_final_checkpoint(self):
    checkpoints_dir = self.models_dir()
    all_dirs = [d for d in checkpoints_dir.rglob('*') if d.is_dir()]
    if not all_dirs: return None
    latest_dir = max(all_dirs, key=lambda x: x.stat().st_ctime)
    return latest_dir

  def train(self):
    my_data = []
    if self.model_name == "colbert-ir/colbertv2.0" or self.model_name == "answerdotai/answerai-colbert-small-v1":
      print("pick a model name")
      return
    elif self.model_path().exists():
      print(f"model already exists at {self.model_path()}")
      return
    for task in fake_data:
      for symbol in task[1]:
        my_data.append((task[0], symbol))
    for task in hvm3_dataset:
      task_str = task.task
      for symbol in task.related_symbols:
        my_data.append((task_str, symbol))
    trainer = RAGTrainer(
        pretrained_model_name="answerdotai/answerai-colbert-small-v1",
        model_name=self.model_name,
        n_usable_gpus=-1,
    )
    trainer.prepare_training_data(
        raw_data=my_data, all_documents=get_all_names()[0])
    trainer.train(
      dim=96
    )
    output_path = self.find_final_checkpoint()
    print(f"trained model_path: {output_path}")
    # Move index to expected location
    self.model_path().parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_path), str(self.model_path()))
    self.rag = RAGPretrainedModel.from_pretrained(str(self.model_path()), verbose=-1)
    return self

  def index_path(self) -> Path:
    return Path(".ragatouille") / "colbert" / "indexes" / self.index_name

  def model_path(self) -> Path:
    if self.model_name == 'colbert-ir/colbertv2.0':
      return 'colbert-ir/colbertv2.0'
    else:
        return Path(".ragatouille") / "colbert" / "retrieval" / self.model_name

  def models_dir(self) -> Path:
    # ragatouille buggy
    return Path(".ragatouille") / "colbert"

  def index_exists(self) -> bool:
    return self.index_path().exists()

  def create_index_if_not_exists(self):
    """Create a persistent index of all planning/action/outcome triplets"""
    from ragatouille import RAGPretrainedModel
    warnings.filterwarnings('ignore')
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
      if self.index_exists():
        p = self.index_path()
        print(f"loading index from {p}")
        self.rag = RAGPretrainedModel.from_index(p, verbose=-1)
        return self.rag

      symbols, _, __ = get_all_names()

      metadatas = [
          {
              "index": i,
              "symbol_name": symbol_name,
          }
          for i, symbol_name in enumerate(symbols)
      ]

      # Create the index
      model = RAGPretrainedModel.from_pretrained(str(self.model_path()), verbose=-1)
      index_path = model.index(
          collection=symbols,
          document_ids=[str(i) for i in range(len(symbols))],
          document_metadatas=metadatas,
          index_name=self.index_name,
          max_document_length=512,
          split_documents=False
      )
      # Move index to expected location
      index_path = Path(index_path)
      if index_path != self.index_path():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
          shutil.move(str(index_path), str(self.index_path()))
    warnings.resetwarnings()
    self.rag = RAGPretrainedModel.from_index(self.index_path(), verbose=-1)
    return self


def run_server():
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)

def print_results(res):
  for result in res:
    print(result['document_metadata']['symbol_name'])
    print(result['score'])
    print('---')

if __name__ == "__main__":
  colbert1 = Colbert(model_name="codebase_symbols_finetune", index_name="codebase_symbols")
#   colbert1.train()
  query = "extend Lam and App nodes to also store a label, just like Sups and Dups. the App-Lam rule must be updated so that, when the labels are different, the nodes will commute instead of beta-reducing"
  res = colbert1.search(query)
  print_results(res)
  print('---')
#   colbert1.train()
  colbert2 = Colbert()
  res = colbert2.search(query)
  print_results(res)