# M5Stack LLM Module + StackFlow `llm_framework` セットアップ手順（Qwen2.5-0.5B + Soft-Prefix 注入）

この README は、M5Stack LLM Module 上で `StackFlow/projects/llm_framework` をビルド・起動し、**Soft-Prefix（prefix 埋め込み）を入力埋め込みの先頭に合成する方式**で注入しつつ、Mac から動作検証するまでの手順をまとめたものです。

> 重要（結論）
> - **ストリーム出力（`llm.utf-8.stream`）は途中で止まるケースがある**ため、まずは **非ストリーム出力（`llm.utf-8`）**で検証します。
> - `setup` が `unit call false (-9)` の場合は **LLMユニット（`llm_llm-*`）が見つからない／起動できない（依存ライブラリ不整合）**の可能性が濃厚です。
> - `ldd` で `GLIBCXX_3.4.30 not found` などが出る場合は、**M5Stack 側の libstdc++ の参照先修正**が必要です。

---

## 0. 前提

- M5Stack LLM Module に SSH で入れること
- `llm-sys` が起動して TCP **10001** を listen していること
- モデルは例として `qwen2.5-0.5B-prefill-20e` を使用（適宜置き換え）

### 0.1 `llm-sys` 稼働確認

```bash
ssh $M5_HOST '
set -e
systemctl status llm-sys --no-pager -l || true
ss -lntp | grep ":10001" || true
'
```

---

## 1. ビルド（`llm_framework`）

### 1.1 ルートへ移動

```bash
ssh $M5_HOST 'cd /root/StackFlow/projects/llm_framework && pwd'
```

### 1.2 （必要なら）`simdjson_component` の `GCC_DUMPMACHINE` KeyError を回避

エラー例：

```
KeyError: 'GCC_DUMPMACHINE'
.../simdjson_component/SConstruct: gcc_dumpmachine = env["GCC_DUMPMACHINE"].split("-")
```

対処（安全に差し込み）：

```bash
ssh $M5_HOST 'python3 -' <<'PY'
from pathlib import Path

p = Path("/root/StackFlow/SDK/components/simdjson_component/SConstruct")
txt = p.read_text()

needle = 'gcc_dumpmachine = env["GCC_DUMPMACHINE"].split("-")'
if needle not in txt:
    print("Pattern not found. Please locate GCC_DUMPMACHINE usage manually.")
    raise SystemExit(1)

marker = "ensure GCC_DUMPMACHINE exists even if SCons GCC tool could not populate it"
if marker in txt:
    print("Already patched.")
    raise SystemExit(0)

patch_lines = [
    "# ensure GCC_DUMPMACHINE exists even if SCons GCC tool could not populate it",
    "import subprocess",
    "if \"GCC_DUMPMACHINE\" not in env:",
    "    try:",
    "        cc = env.get(\"CC\", \"gcc\")",
    "        dm = subprocess.check_output([cc, \"-dumpmachine\"]).decode().strip()",
    "    except Exception:",
    "        dm = \"aarch64-linux-gnu\"",
    "    env[\"GCC_DUMPMACHINE\"] = dm",
    "",
]
patch = "\n".join(patch_lines)

txt = txt.replace(needle, patch + "\n" + needle)
p.write_text(txt)
print("Patched:", p)
PY
```

---

### 1.3 （必要なら）SCons が参照するクロスツールチェーンの “ダミーラッパー” を作る

SCons が以下のような **存在しないクロスコンパイラ**を呼びに行くことがあります：

```
/opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
```

M5Stack（aarch64 実機）では、そのパスに **ネイティブ gcc/g++ を呼ぶラッパー**を置くのが最短です。

```bash
ssh $M5_HOST '
set -e
TC=/opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin
PREFIX=aarch64-none-linux-gnu

mkdir -p "$TC"
cd "$TC"

cat > ${PREFIX}-gcc << "SH"
#!/bin/sh
exec /usr/bin/gcc "$@"
SH

cat > ${PREFIX}-g++ << "SH"
#!/bin/sh
exec /usr/bin/g++ "$@"
SH

chmod +x ${PREFIX}-gcc ${PREFIX}-g++

# binutils系（必要になることがある）
for t in ar ranlib strip nm ld objcopy objdump; do
  if command -v $t >/dev/null 2>&1; then
    ln -sf "$(command -v $t)" "${PREFIX}-${t}"
  fi
done
'
```

---

### 1.4 ビルド（OOM回避のため `-j1` 推奨）

ビルド中にダウンロード確認（Y/N）が出るため、`yes Y |` で全て自動承諾します。

```bash
ssh $M5_HOST '
set -e
cd /root/StackFlow/projects/llm_framework
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
yes Y | scons -j1
'
```

---

## 2. `setup` が `unit call false (-9)` の場合（最重要トラブルシュート）

### 2.1 症状

クライアントの `setup` 応答が以下になる：

```json
{"error":{"code":-9,"message":"unit call false"}}
```

これは多くの場合、**LLMユニット（`llm_llm-*`）が起動できない**（= 依存ライブラリが足りない/ABI不整合）ことが原因です。

### 2.2 まず `ldd` で依存関係を確認

```bash
ssh $M5_HOST '
set -e
ldd /opt/m5stack/bin/llm_llm-1.9 | egrep "GLIBCXX|CXXABI|not found" || echo OK
'
```

よくある失敗：
- `GLIBCXX_3.4.30 not found`
- `CXXABI_1.3.13 not found`
- `libax_interpreter.so => not found`

---

## 3. 依存関係（libstdc++）の修復：`GLIBCXX_3.4.30 not found` 対策

### 3.1 システム側に正しい libstdc++ があるか確認

Ubuntu 22.04 の場合、多くはここにあります：

- `/usr/lib/aarch64-linux-gnu/libstdc++.so.6`（`GLIBCXX_3.4.30` を含む）

確認：

```bash
ssh $M5_HOST '
set -e
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | egrep "GLIBCXX_3.4.30|CXXABI_1.3.13" | head
'
```

### 3.2 M5Stack 独自パス側（例：`/usr/local/m5stack/lib/gcc-10.3/`）を壊さずに直す

`cp -a` で symlink をコピーすると **壊れたリンク**になることがあるため、以下のように **絶対パス symlink を貼り直す**のが安全です。

```bash
ssh $M5_HOST '
set -e
DIR=/usr/local/m5stack/lib/gcc-10.3
cd "$DIR"
TS=$(date +%Y%m%d_%H%M%S)

# 既存（壊れている可能性あり）を退避
if [ -L libstdc++.so.6 ] || [ -e libstdc++.so.6 ]; then
  mv -f libstdc++.so.6 "libstdc++.so.6.broken_$TS" || true
fi

# システム側へ向ける
ln -sfn /usr/lib/aarch64-linux-gnu/libstdc++.so.6 libstdc++.so.6

# 確認
ls -l libstdc++.so.6*
strings -a libstdc++.so.6 | egrep "GLIBCXX_3.4.30|CXXABI_1.3.13" | head -n 5
'
```

---

## 4. 依存関係（Axera runtime）修復：`libax_interpreter.so not found` 対策

### 4.1 まず場所を探す

```bash
ssh $M5_HOST '
set -e
find /opt /usr/local/m5stack /usr/lib -name "libax_interpreter.so*" 2>/dev/null | head -n 50
'
```

### 4.2 見つかったディレクトリを `ldconfig` に登録

例：`/usr/local/m5stack/lib/ax-lib` にあった場合

```bash
ssh $M5_HOST '
set -e
DIR=/usr/local/m5stack/lib/ax-lib
ls -l "$DIR"/libax_interpreter.so* >/dev/null

echo "$DIR" > /etc/ld.so.conf.d/m5stack-ax.conf
ldconfig
ldconfig -p | grep -i ax_interpreter || true
'
```

---

## 5. baseline 動作確認（非ストリーム）

ストリーム出力は途中で止まるケースがあったため、まずは **非ストリーム**で確認します。

### 5.1 baseline（non-stream）確認（例）

- `response_format: "llm.utf-8"`
- `object: "llm.utf-8"`

を使い、推論結果を 1発で返します。

> 会話ログ上、この方法では **最後まで応答が返る**ことを確認済みです。

---

## 6. Soft-Prefix 注入（非ストリーム出力で比較）

### 6.1 なぜ “入力は stream、出力は non-stream” なのか

- 出力ストリーム（`llm.utf-8.stream`）が途中停止することがある
- Soft-Prefix を JSON `data` 内に同梱するため、入力は `llm.utf-8.stream` 形式（`delta/index/finish`）を使うのが都合が良い
- 出力は `llm.utf-8` にして、最終結果を 1回で受け取る

### 6.2 Soft-Prefix 形式（本READMEでの前提）

- `soft_prefix.len = P`（prefix token 数）
- `soft_prefix.data_b64` は **bf16(u16 little-endian)** の配列を base64 したもの
- 要素数は **`P * H`**
  - `P` = prefix token 数（例：1）
  - `H` = `tokens_embed_size`（例：896。モデル設定に依存）

### 6.3 baseline vs soft_prefix 比較スクリプト（非ストリーム出力）

このリポジトリに含める想定の比較スクリプト例：

- `compare_baseline_vs_softprefix_nostream.py`

機能：
- baseline（prefixなし）を1回実行して保存
- prefixあり（val=小→大）を複数回実行
- 類似度/差分（先頭）/全文（JSON）を保存

---

## 7. モデルの `tokens_embed_size (H)` を確認する

環境例ではモデルJSONが以下にありました：

- `/opt/m5stack/data/models/mode_qwen2.5-0.5B-prefill-20e.json`
- `/opt/m5stack/data/qwen2.5-0.5B-prefill-20e/qwen2.5-0.5B-prefill-20e.json`

確認コマンド例：

```bash
ssh $M5_HOST '
set -e
grep -n ""tokens_embed_size"" /opt/m5stack/data/models/mode_qwen2.5-0.5B-prefill-20e.json | head -n 5 || true
'
```

---

## 8. よくあるエラーと対処まとめ

### 8.1 `unit call false (-9)`
- LLMユニットが起動できていない（依存関係）
- `ldd /opt/m5stack/bin/llm_llm-*` を確認し、`GLIBCXX`/`libax_interpreter` を解消

### 8.2 `inference data push false (-4)`
- setup が失敗してタスクが無い／入力形式が合っていないケースで出やすい
- まず `setup` を code=0 にする（依存関係の解消が前提）

### 8.3 ストリーム出力が途中で止まる
- 非ストリーム（`llm.utf-8`）で検証する
- ストリームを直す場合は、LLM側の callback/decode頻度・tokenizer 経路の見直しが必要

---

## 9. 注意事項

- `/usr/local/m5stack/lib/gcc-10.3/libstdc++.so.6` のような **システム寄りの領域**を書き換える場合は、必ずバックアップを取ってください。
- `ldconfig` 変更はシステム全体に影響します。影響範囲を理解した上で実施してください。
- soft-prefix の注入が効いていない（出力が baseline と完全一致）場合は、LLMユニット側で **prefix合成が有効になっているか**（JSON受理→bf16復元→埋め込み先頭に合成）を確認してください。

---

## License
この README 自体は自由に改変・再配布してOKです（プロジェクト本体のライセンスに従ってください）。
