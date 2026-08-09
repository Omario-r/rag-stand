"""Нарезка корпуса на чанки: размеры, покрытие золотого набора, глазами.

Пункт 4 этапа 3 из роадмапа — «глазами проверить 20 случайных чанков» —
не пропускать: полчаса чтения экономят неделю объяснений, почему поиск
странно себя ведёт.

Запуск:  uv run python scripts/check-chunks.py [сколько показать]
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.chunk import MAX_TOKENS, chunk_document, estimate_tokens, parse  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
DOCS = {
    "TR-114-2026.md": ("TR-114", "перевозка"),
    "EKS-77-2026.md": ("EKS-77", "экспедирование"),
    "AR-203-2026.md": ("AR-203", "аренда_ТС"),
    "TR-114-2026-DS1.md": ("TR-114-DS1", "допсоглашение"),
    "SH-45-2026.docx": ("SH-45", "хранение"),
}

chunks = []
for name, (doc_id, doc_type) in DOCS.items():
    title, clauses = parse(CORPUS / name)
    got = chunk_document(doc_id, title, clauses, doc_type=doc_type)
    chunks += got
    sizes = sorted(estimate_tokens(c.text) for c in got)
    merged = sum(1 for c in got if len(c.clause_ids) > 1)
    print(f"{doc_id:11s} {len(clauses):3d} пунктов -> {len(got):3d} чанков "
          f"(склеено {merged})  оценка: мед {sizes[len(sizes)//2]:3d}  max {sizes[-1]:3d}")

sizes = sorted(estimate_tokens(c.text) for c in chunks)
n = len(sizes)
# ЭТО ОЦЕНКА, А НЕ ЗАМЕР: len/CHARS_PER_TOKEN_WORST, завышает примерно
# в 1,4 раза у Qwen3 и вдвое у e5. Настоящие числа — scripts/measure-tokens.py,
# он ходит в /tokenize работающего сервера и потому требует поднятого стенда.
print(f"\nвсего чанков: {n}   ОЦЕНКА токенов: мин {sizes[0]}  мед {sizes[n//2]}  "
      f"p90 {sizes[int(n*0.9)]}  max {sizes[-1]}")
print(f"короче 40 токенов: {sum(1 for s in sizes if s < 40)}   "
      f"длиннее {MAX_TOKENS}: {sum(1 for s in sizes if s > MAX_TOKENS)}")

# --- сверка с золотым набором: ссылка может указывать на подпункт внутри чанка
gs = yaml.safe_load((CORPUS / "golden-set.yaml").read_text(encoding="utf-8"))
refs = set()
for q in gs["questions"]:
    refs |= set(q.get("relevant", [])) | set(q.get("superseded", []))

covered = set()
for ch in chunks:
    doc = ch.id.split("#")[0]
    covered |= {f"{doc}#{cid}" for cid in ch.clause_ids}
missing = sorted(refs - covered)
print(f"\nссылок в наборе: {len(refs)}   "
      + (f"НЕ ПОКРЫТО: {missing}" if missing else "все покрыты чанками"))

# --- связи допсоглашения
links = [(ch.id, a) for ch in chunks for a in ch.payload["amends"]]
print(f"\nсвязи допсоглашения ({len(links)}):")
base = {cid.split("#")[1] for ch in chunks for cid in [ch.id] if ch.id.startswith("TR-114#")}
for cid, a in links:
    exists = a["пункт"] in base
    note = "" if exists else "  (в основном договоре такого пункта нет — сходится с «добавляет»)"
    print(f"  {cid}  {a['операция']:10s} TR-114#{a['пункт']}{note}")

print("\n=== самые короткие чанки: осмысленны ли поодиночке ===")
for ch in sorted(chunks, key=lambda c: estimate_tokens(c.text))[:6]:
    print(f"  ~{estimate_tokens(ch.text):3d} ток.  {ch.id:16s} {ch.text[:70]}")

# --- глазами
k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print(f"\n=== {k} чанков целиком (то, что уйдёт в вектор) ===")
step = max(1, len(chunks) // k)
for ch in chunks[::step][:k]:
    print(f"\n--- {ch.id}  пункты={ch.clause_ids}  "
          f"~{estimate_tokens(ch.text)} ток. (оценка) ---")
    print(ch.embed_text[:600])
