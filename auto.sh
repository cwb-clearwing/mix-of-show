#!/bin/bash

# 后台任务管理脚本

if [ $# -eq 0 ]; then
    echo "使用方法: $0 '要执行的命令'"
    echo "示例: $0 'while true; do date; sleep 5; done'"
    exit 1
fi

# 获取脚本所在绝对路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 生成安全文件名（替换特殊字符）
SAFE_CMD_NAME=$(echo "$1" | tr ' ' '_' | tr -cd '[:alnum:]._-' | cut -c -40)
OUTPUT_FILE="${SCRIPT_DIR}/${TIMESTAMP}_${SAFE_CMD_NAME}.log"

# 执行命令并捕获输出
{
    echo "▶️ 开始执行命令: $1"
    echo "🕒 启动时间: $(date +'%Y-%m-%d %H:%M:%S')"
    echo "📁 输出文件: ${OUTPUT_FILE}"
    echo "--------------------------------------------------"
    eval "$1"
} > "$OUTPUT_FILE" 2>&1 &

# 获取任务信息
PID=$!
echo "✅ 命令已后台执行"
echo "🔗 PID:       $PID"
echo "📄 输出文件: $OUTPUT_FILE"
echo "💡 查看输出: tail -f \"$OUTPUT_FILE\""

# 记录任务信息到临时文件（可选）
echo "$PID:$1:${OUTPUT_FILE}" >> "${SCRIPT_DIR}/background_jobs.log"
