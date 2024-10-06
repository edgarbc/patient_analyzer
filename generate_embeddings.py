from transformers import AutoTokenizer, AutoModel
import torch


def load_embeddings_model():
    """
    Load a pre-trained medical BERT model
    """
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def embed_text(text, tokenizer, model):
    """
    Embed a text using a pre-trained medical BERT model
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings to get a fixed-size representation
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


# code to test the function
if __name__ == "__main__":
    # Load the embeddings model
    tokenizer, model = load_embeddings_model()

    # Compute embeddings for all patients
    #patient_embeddings = [embed_text(text, tokenizer, model) for text in patient_texts]
