#!/bin/bash
# ============================================================
# CRAFT Full Experiment Runner
# Settings: --oracle --oracle_n 5 --no_tools --turns 20
# Runs all API frontier models, then all local open-weight models
# ============================================================

set -e  # exit on error
COMMON="--oracle --oracle_n 5 --no_tools --turns 20"

# ── API Frontier Models ───────────────────────────────────────
API_MODELS=(
    "gpt-4o-mini"
    "gpt-4.1-mini"
    "gpt-4o"
    "gemini-2.5-flash"
    "gemini-2.5-flash-lite"
    "gemini-3-flash-preview"
    "gemini-3.1-flash-lite-preview"
    "claude-haiku-4-5"
    "claude-sonnet-4-6"
)

# ── Local Open-Weight Models ──────────────────────────────────
LOCAL_MODELS=(
    "qwen-7b"
    "qwen-14b"
    "qwen-32b"
    "llama-8b"
    "mistral-7b"
    "gemma-9b"
    "deepseek-v2-lite"
    "qwen-72b"
)

# ── Large models that need 4bit quantization ──────────────────
QUANTIZE_4BIT=("qwen-72b" "qwen-32b")

needs_quantize() {
    local model=$1
    for m in "${QUANTIZE_4BIT[@]}"; do
        [[ "$m" == "$model" ]] && return 0
    done
    return 1
}

echo "============================================================"
echo " CRAFT Full Experiment Runner"
echo " Settings: $COMMON"
echo " $(date)"
echo "============================================================"

# ── Run API models ────────────────────────────────────────────
echo ""
echo ">>> Running API frontier models (${#API_MODELS[@]} total)"
echo "------------------------------------------------------------"

for model in "${API_MODELS[@]}"; do
    echo ""
    echo "[API] Starting: $model  @ $(date '+%H:%M:%S')"
    python run_craft.py \
        --mode api \
        --director "$model" \
        $COMMON \
        && echo "[API] Done: $model" \
        || echo "[API] FAILED: $model — continuing"
done

# ── Run local models ──────────────────────────────────────────
echo ""
echo ">>> Running local open-weight models (${#LOCAL_MODELS[@]} total)"
echo "------------------------------------------------------------"

for model in "${LOCAL_MODELS[@]}"; do
    echo ""
    echo "[LOCAL] Starting: $model  @ $(date '+%H:%M:%S')"

    if needs_quantize "$model"; then
        echo "[LOCAL] Using 4bit quantization for $model"
        python run_craft.py \
            --mode local \
            --director "$model" \
            --quantize 4bit \
            $COMMON \
            && echo "[LOCAL] Done: $model" \
            || echo "[LOCAL] FAILED: $model — continuing"
    else
        python run_craft.py \
            --mode local \
            --director "$model" \
            $COMMON \
            && echo "[LOCAL] Done: $model" \
            || echo "[LOCAL] FAILED: $model — continuing"
    fi
done

echo ""
echo "============================================================"
echo " All models complete @ $(date)"
echo "============================================================"