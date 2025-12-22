# BI_M5_QwenSoftPrefix — M5Stack LLM Module（StackFlow `llm_framework`）セットアップ手順  
（Gitで管理 → `scp`で `main.cpp` / `LLM.hpp` を転送 → ビルド → Macから動作検証）

この README は、M5Stack LLM Module（AX650/AX630系）上の `StackFlow/projects/llm_framework` を対象に、

- **Mac側で Git 管理**（差分が追える）
- **Mac → M5Stackへ `scp` で `main.cpp` / `LLM.hpp` を安全に置き換え**
- **M5Stack側で `scons` ビルド**
- **Mac側からソケットAPIで動作検証（非ストリーム推奨）**
- **Soft-Prefix（bf16ベクトル）注入の比較検証**

までを、最初から最後まで「手順通りに打てる」形でまとめたものです。

---

## 0. 前提

- Mac（ローカル）に `git`, `python3`, `ssh`, `scp` がある
- M5Stackに `root` で SSH 接続できる（`$M5_HOST` で参照）
- M5Stack で `llm-sys` が `10001/tcp` を listen している

### 0.1 接続先を環境変数に入れる（Mac）

```bash
export M5_HOST="root@192.168.3.132"   # ←あなたのM5のIPに合わせる
```

### 0.2 llm-sys 稼働確認（Mac→SSH）

```bash
ssh $M5_HOST '
set -e
systemctl status llm-sys --no-pager -l || true
ss -lntp | grep ":10001" || true
'
```

---

## 1. Git リポジトリを作る（Mac）

### 1.1 作業ディレクトリ作成 & 初期化

```bash
mkdir -p ~/CODES/BI_M5_QwenSoftPrefix
cd ~/CODES/BI_M5_QwenSoftPrefix

git init
```

### 1.2 このリポジトリの推奨構成

```
BI_M5_QwenSoftPrefix/
  README.md
  patched/
    main.cpp
    LLM.hpp
  scripts/
    client_baseline_nostream.py
    client_two_infer_nostream.py
    compare_baseline_vs_softprefix_nostream.py
```

> `patched/` に「M5へ転送する最終版」を置いておくのが運用上ラクです（現在どれを使っているか迷子にならない）。

---

## 2. M5Stackから “現状ファイル” を取ってコミット（最重要）

まず **現状（動いている状態）をローカルに回収してGitで保存**します。  
これがあると、何か壊しても **いつでも即ロールバック**できます。

### 2.1 M5Stack上の対象パス

- `main.cpp`：`/root/StackFlow/projects/llm_framework/main_llm/src/main.cpp`
- `LLM.hpp`：`/root/StackFlow/projects/llm_framework/main_llm/src/runner/LLM.hpp`

### 2.2 `scp` で回収（Mac）

```bash
mkdir -p patched/original

scp $M5_HOST:/root/StackFlow/projects/llm_framework/main_llm/src/main.cpp patched/original/main.cpp
scp $M5_HOST:/root/StackFlow/projects/llm_framework/main_llm/src/runner/LLM.hpp patched/original/LLM.hpp
```

### 2.3 “オリジナル”をコミット

```bash
git add patched/original/main.cpp patched/original/LLM.hpp
git commit -m "backup: original main.cpp and LLM.hpp from device"
```

---

## 3. 変更（パッチ）を作る／反映する（Mac）

あなたが編集するのは基本この2つです：

- `patched/main.cpp`
- `patched/LLM.hpp`

> ここに「Soft-Prefix注入」「2回目タイムアウト対策（prefill入力の完全初期化）」などの修正を入れます。

### 3.1 “2回目以降にタイムアウトする”対策（重要）

症状：再起動後は1回目OKだが、2回目同じ推論を投げると止まる。  
原因候補：**prefill固定長入力の未初期化領域が前回の値を引きずる**。

修正方針（`LLM.hpp`）：

- `Run(test_embed)` の prefill入力を毎回 `prefill_token_num * H` へ `resize(..., 0)` で **ゼロ埋め**
- `indices` を `0..prefill_token_num-1` まで必ず **全要素埋め**

> これで「2回目だけハング/超遅延」が改善するケースが多いです。

### 3.2 編集したらコミット

```bash
# 例：編集後に
git add patched/main.cpp patched/LLM.hpp
git commit -m "feat: soft-prefix injection + fix 2nd-run timeout (prefill input init)"
```

### 3.3 GitHubへ上げる（任意）

```bash
git remote add origin git@github.com:YOUR_NAME/BI_M5_QwenSoftPrefix.git
git branch -M main
git push -u origin main
```

---

## 4. Mac → M5Stackへ `scp` で転送して置き換え（安全手順）

**直接上書きは避け**、まず `/tmp` に転送してから、M5上でバックアップしつつ入れ替えます。

### 4.1 転送（Mac）

```bash
scp patched/main.cpp $M5_HOST:/tmp/main.cpp.new
scp patched/LLM.hpp  $M5_HOST:/tmp/LLM.hpp.new
```

### 4.2 バックアップ & 置換（M5側）

```bash
ssh $M5_HOST '
set -e
PROJ=/root/StackFlow/projects/llm_framework
TS=$(date +%Y%m%d_%H%M%S)

# backup
cp -a $PROJ/main_llm/src/main.cpp $PROJ/main_llm/src/main.cpp.bak_$TS
cp -a $PROJ/main_llm/src/runner/LLM.hpp $PROJ/main_llm/src/runner/LLM.hpp.bak_$TS

# replace
install -m 644 /tmp/main.cpp.new $PROJ/main_llm/src/main.cpp
install -m 644 /tmp/LLM.hpp.new  $PROJ/main_llm/src/runner/LLM.hpp

echo "replaced OK"
ls -lh $PROJ/main_llm/src/main.cpp $PROJ/main_llm/src/runner/LLM.hpp
'
```

---

## 5. M5Stack側でビルド（`scons`）

### 5.1 `simdjson_component` の `GCC_DUMPMACHINE` KeyError が出る場合（初回だけ）

エラー例：
```
KeyError: 'GCC_DUMPMACHINE'
... simdjson_component/SConstruct ...
```

パッチ（M5側で実行）：

```bash
ssh $M5_HOST 'python3 -' <<'PY'
from pathlib import Path
p = Path("/root/StackFlow/SDK/components/simdjson_component/SConstruct")
txt = p.read_text()
needle = 'gcc_dumpmachine = env["GCC_DUMPMACHINE"].split("-")'
marker = "ensure GCC_DUMPMACHINE exists even if SCons GCC tool could not populate it"
if needle not in txt:
    raise SystemExit("Pattern not found.")
if marker in txt:
    print("Already patched.")
    raise SystemExit(0)
patch = "\n".join([
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
])
txt = txt.replace(needle, patch + "\n" + needle)
p.write_text(txt)
print("Patched:", p)
PY
```

### 5.2 SCons が “存在しないクロスコンパイラ” を呼ぶ場合（初回だけ）

例：
```
/opt/gcc-arm-10.3.../aarch64-none-linux-gnu-g++: No such file or directory
```

M5は aarch64 実機なので、そこに **gcc/g++ラッパー**を置きます：

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
for t in ar ranlib strip nm ld objcopy objdump; do
  if command -v $t >/dev/null 2>&1; then
    ln -sf "$(command -v $t)" "${PREFIX}-${t}"
  fi
done
echo "wrappers installed"
'
```

### 5.3 ビルド（OOM回避で `-j1` 推奨）

```bash
ssh $M5_HOST '
set -e
cd /root/StackFlow/projects/llm_framework
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
yes Y | scons -j1
'
```

---

## 6. 生成物（`llm_llm-*`）を導入して依存関係チェック

### 6.1 生成物を /opt/m5stack/bin へ

```bash
ssh $M5_HOST '
set -e
NEW=/root/StackFlow/projects/llm_framework/build/llm_llm-1.8/llm_llm-1.8
DEST=/opt/m5stack/bin/llm_llm-1.8
TS=$(date +%Y%m%d_%H%M%S)

ls -lh "$NEW"

if [ -f "$DEST" ]; then
  cp "$DEST" "$DEST.bak_$TS"
fi
install -m 755 "$NEW" "$DEST"

ln -sfn "$DEST" /opt/m5stack/bin/llm_llm
ln -sfn "$DEST" /opt/m5stack/bin/llm-llm

echo "installed:"
ls -lh /opt/m5stack/bin/llm_llm /opt/m5stack/bin/llm-llm /opt/m5stack/bin/llm_llm-1.8
'
```

### 6.2 `ldd` で “not found / GLIBCXX” を潰す

```bash
ssh $M5_HOST '
set -e
ldd /opt/m5stack/bin/llm_llm-1.8 | egrep "not found|GLIBCXX|CXXABI" || echo "OK"
'
```

#### 6.2.1 `GLIBCXX_3.4.30 not found` が出る場合（例）

M5独自の `libstdc++.so.6` が古い/壊れている可能性があるため、  
**システム側（Ubuntu）の libstdc++ を参照する symlink に貼り替え**ます。

```bash
ssh $M5_HOST '
set -e
DIR=/usr/local/m5stack/lib/gcc-10.3
cd "$DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ -L libstdc++.so.6 ] || [ -e libstdc++.so.6 ]; then
  mv -f libstdc++.so.6 "libstdc++.so.6.broken_$TS" || true
fi

ln -sfn /usr/lib/aarch64-linux-gnu/libstdc++.so.6 libstdc++.so.6

ls -l libstdc++.so.6*
strings -a libstdc++.so.6 | egrep "GLIBCXX_3.4.30|CXXABI_1.3.13" | head -n 5
'
```

#### 6.2.2 `libax_interpreter.so => not found` が出る場合

まず場所を探す：

```bash
ssh $M5_HOST '
set -e
find /opt /usr/local/m5stack /usr/lib -name "libax_interpreter.so*" 2>/dev/null | head -n 50
'
```

見つかったディレクトリ（例：`/usr/local/m5stack/lib/ax-lib`）を `ldconfig` へ登録：

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

## 7. サービス再起動

```bash
ssh $M5_HOST '
set -e
systemctl restart llm-sys
sleep 1
ss -lntp | grep ":10001" || true
'
```

---

## 8. Macから動作検証（非ストリーム推奨）

ストリーム（`llm.utf-8.stream`）は途中停止するケースがあるため、まず **非ストリーム**で通します。

### 8.1 baseline（非ストリーム）

`scripts/client_baseline_nostream.py` を用意して実行します。

チェックポイント：
- `setup` の `error.code == 0`
- `inference` の応答が **1回で返る**

### 8.2 “2回連続推論”で再現しないか確認（重要）

`scripts/client_two_infer_nostream.py` を使い、  
**同一の work_id に対して inference を2回連続**で投げて通るか確認します。

---

## 9. Soft-Prefix 注入の比較（非ストリーム出力で）

### 9.1 方針（推奨）

- 入力：`llm.utf-8.stream`（dataに `delta/index/finish` を含めやすい）
- 出力：`llm.utf-8`（一括で受け取って比較しやすい）

### 9.2 比較スクリプト

`scripts/compare_baseline_vs_softprefix_nostream.py` で、
- baseline（prefixなし）
- prefixあり（val小→大）
を複数回回して、応答テキストを保存・比較します。

---

## 10. よくあるエラーと原因

### `unit call false (-9)`（setup失敗）
- `llm_llm-*` が起動できない（依存ライブラリ不整合）
- `ldd /opt/m5stack/bin/llm_llm-*` を見て `GLIBCXX / libax_interpreter` を解消

### `inference data push false (-4)`
- setup が通っていない / 入力形式が合っていない / タスクが作れてない

### 2回目だけタイムアウト
- prefill固定長入力の未初期化や状態引きずりが原因になりやすい  
  → `LLM.hpp` で **prefill入力のゼロ埋め**、`indices` の全埋めを必ず行う

---

## 11. 注意

- `/usr/local/m5stack/lib/...` 配下の変更はシステム全体に影響します。必ずバックアップを取ってください。
- まずは **非ストリーム**で安定動作を確認してから、ストリーム再挑戦をおすすめします。

---

## License
この README は自由に改変・再配布してOKです（プロジェクト本体のライセンスに従ってください）。
