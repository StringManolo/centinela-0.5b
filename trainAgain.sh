# 1. Añadir datos nuevos a data/train.jsonl (append, no tocar los viejos)
#    Ejemplo: generar 40 pares más por clase nueva
python3 scripts/gen_extended_data.py --n-pairs 40 --append data/train.jsonl
# Better to add quality training examples


# 2. Commit + push
git add data/train.jsonl
git commit -m "add more training data"
git push

# 3. Lanzar goteo (resume_from=latest automáticamente)
gh workflow run centinela-optimized-training.yml -f resume_from=latest
