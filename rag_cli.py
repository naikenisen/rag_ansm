"""
RAG CLI - Client terminal pour interroger la base documentaire ANSM
"""

import sys
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration des chemins
DOCUMENTS_PATH = Path(__file__).parent / "base_documentaire"
PERSIST_DIR = Path(__file__).parent / "storage"
MODELS_CACHE = Path(__file__).parent / "models_cache"

# Template de requête
QA_PROMPT = PromptTemplate(
    """Tu es un assistant expert sur la base de données des médicaments (BDPM) de l'ANSM.

Contexte extrait des documents :
---------------------
{context_str}
---------------------

Question : {query_str}

Instructions :
- Base ta réponse principalement sur le contexte fourni ci-dessus.
- Tu peux utiliser tes connaissances générales pour expliquer des termes médicaux ou pharmaceutiques.
- Si le contexte ne contient pas l'information demandée, indique-le clairement.
- Réponds en français de manière claire et structurée.

Réponse :"""
)


def print_status(message: str):
    """Affiche un message de statut."""
    print(f"\033[94m[INFO]\033[0m {message}")


def print_error(message: str):
    """Affiche un message d'erreur."""
    print(f"\033[91m[ERREUR]\033[0m {message}")


def print_success(message: str):
    """Affiche un message de succès."""
    print(f"\033[92m[OK]\033[0m {message}")


def configure_settings():
    """Configure LlamaIndex avec les modèles."""
    print_status("Chargement du modèle d'embedding...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        cache_folder=str(MODELS_CACHE),
        embed_batch_size=32,
    )
    
    print_status("Connexion au LLM Ollama (qwen2.5:3b)...")
    Settings.llm = Ollama(
        model="qwen2.5:3b",
        request_timeout=180.0,
        temperature=0.1,
        context_window=4096,
    )
    
    Settings.chunk_size = 512
    Settings.chunk_overlap = 100


# Mapping des fichiers vers leurs descriptions
FILE_DESCRIPTIONS = {
    "CIS_bdpm.txt": "MÉDICAMENTS - Liste des médicaments avec dénomination, forme pharmaceutique, voie d'administration, statut AMM, laboratoire",
    "CIS_CIP_bdpm.txt": "PRÉSENTATIONS - Codes CIP des présentations, libellés, statut de commercialisation, prix",
    "CIS_COMPO_bdpm.txt": "COMPOSITION - Substances actives des médicaments avec dosages (paracétamol, ibuprofène, etc.)",
    "CIS_GENER_bdpm.txt": "GÉNÉRIQUES - Groupes génériques, médicaments princeps et leurs génériques",
    "CIS_HAS_SMR_bdpm.txt": "AVIS SMR HAS - Service Médical Rendu (Important, Modéré, Faible, Insuffisant) par indication thérapeutique",
    "CIS_HAS_ASMR_bdpm.txt": "AVIS ASMR HAS - Amélioration du Service Médical Rendu (I à V)",
    "CIS_CPD_bdpm.txt": "CONDITIONS PRESCRIPTION - Conditions de prescription et de délivrance",
    "CIS_CIP_Dispo_Spec.txt": "DISPONIBILITÉ - Ruptures de stock et tensions d'approvisionnement",
    "CIS_InfoImportantes": "INFORMATIONS IMPORTANTES - Alertes et informations de sécurité",
    "HAS_LiensPageCT_bdpm.txt": "LIENS HAS - Liens vers les avis de la Commission de Transparence",
    "CIS_MITM.txt": "MITM - Médicaments d'intérêt thérapeutique majeur",
}


def load_documents_with_encoding():
    """Charge les documents en gérant l'encodage ISO-8859/UTF-8."""
    documents = []
    
    for file_path in DOCUMENTS_PATH.glob("*.txt"):
        # Essayer différents encodages
        content = None
        for encoding in ['utf-8', 'iso-8859-1', 'cp1252']:
            try:
                content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print_error(f"Impossible de lire {file_path.name}")
            continue
        
        # Ajouter un en-tête descriptif pour aider le RAG
        file_name = file_path.name
        description = FILE_DESCRIPTIONS.get(file_name, "Fichier de la base BDPM ANSM")
        
        # Préfixer le contenu avec des métadonnées
        header = f"""=== FICHIER: {file_name} ===
DESCRIPTION: {description}
=== CONTENU ===
"""
        enriched_content = header + content
        
        doc = Document(
            text=enriched_content,
            metadata={
                "file_name": file_name,
                "file_path": str(file_path),
                "description": description,
            }
        )
        documents.append(doc)
        print_status(f"Chargé: {file_name} ({len(content):,} caractères)")
    
    return documents


def load_or_create_index(force_reload: bool = False) -> VectorStoreIndex:
    """Charge ou crée l'index vectoriel."""
    
    # Charger l'index existant
    if PERSIST_DIR.exists() and not force_reload:
        try:
            print_status("Chargement de l'index existant...")
            storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
            index = load_index_from_storage(storage_context)
            print_success("Index chargé.")
            return index
        except Exception as e:
            print_error(f"Impossible de charger l'index: {e}")
    
    # Créer un nouvel index
    print_status(f"Indexation des documents depuis {DOCUMENTS_PATH}...")
    
    if not DOCUMENTS_PATH.exists():
        print_error(f"Le dossier {DOCUMENTS_PATH} n'existe pas!")
        sys.exit(1)
    
    # Charger les documents avec gestion de l'encodage
    documents = load_documents_with_encoding()
    
    if not documents:
        print_error("Aucun document .txt trouvé!")
        sys.exit(1)
    
    print_status(f"Création de l'index avec {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    # Sauvegarder
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print_success("Index créé et sauvegardé.")
    
    return index


def reindex():
    """Force la réindexation."""
    import shutil
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    return load_or_create_index(force_reload=True)


def main():
    """Boucle principale du chat."""
    print("\n" + "="*60)
    print("RAG ANSM - Base des Médicaments (Terminal)")
    print("="*60)
    print("Commandes: 'quit' pour quitter, 'reindex' pour réindexer\n")
    
    # Initialisation
    configure_settings()
    index = load_or_create_index()
    
    # Créer le moteur de requête
    query_engine = index.as_query_engine(
        similarity_top_k=8,
        streaming=False,
        text_qa_template=QA_PROMPT,
        response_mode="compact",
    )
    
    print_success("Prêt! Posez vos questions.\n")
    
    # Boucle de chat
    while True:
        try:
            question = input("\033[96m❯ Vous:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAu revoir!")
            break
        
        if not question:
            continue
        
        if question.lower() == "quit":
            print("Au revoir!")
            break
        
        if question.lower() == "reindex":
            index = reindex()
            query_engine = index.as_query_engine(
                similarity_top_k=8,
                streaming=False,
                text_qa_template=QA_PROMPT,
                response_mode="compact",
            )
            continue
        
        # Requête
        print("\033[93m Recherche en cours...\033[0m")
        try:
            response = query_engine.query(question)
            print(f"\n\033[92m Assistant:\033[0m\n{response}\n")
        except Exception as e:
            print_error(f"Erreur: {e}\n")


if __name__ == "__main__":
    main()
