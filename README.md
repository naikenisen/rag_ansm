# RAG ANSM - Base des Médicaments

## précautions d'utilisation
Attention ce logiciel est expérimental et ne doit pas être utilisé à des fins médicales.

## Voici un exemple d'utilisation de ce RAG : 

```bash
============================================================
RAG ANSM - Base des Médicaments (Terminal)
============================================================
Commandes: 'quit' pour quitter, 'reindex' pour réindexer

[INFO] Chargement du modèle d'embedding...
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3153.31it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: intfloat/multilingual-e5-small
Key                     | Status     | Details
------------------------+------------+--------
embeddings.position_ids | UNEXPECTED |        

Notes:
- UNEXPECTED          :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
[INFO] Connexion au LLM Ollama (qwen2.5:3b)...
[INFO] Chargement de l'index existant...
[OK] Index chargé.
[OK] Prêt! Posez vos questions.

❯ Vous: est ce que le fentanyl est disponible au dosage 65µg en france
 Recherche en cours...

 Assistant:
Selon les informations fournies dans le contexte fourni, il n'y a aucune indication que le fentanyl est disponible au dosage précis de 65 µg (microgrammes) en France. Le contexte mentionne plusieurs dosages différents du fentanyl, notamment :

- Un dosage de 10 µg
- Des doses allant jusqu'à 200 µg pour certaines utilisations orales
- Des doses de 30 µg et 50 µg

Cependant, il n'y a pas de référence directe au dosage exact de 65 µg. Il est possible que ce dosage ne soit pas officiellement disponible en France ou qu'il ne soit pas largement utilisé dans les pratiques médicales courantes.

❯ Vous: 

Au revoir!
```