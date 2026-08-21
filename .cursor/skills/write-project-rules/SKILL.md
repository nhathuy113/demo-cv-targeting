---
name: write-project-rules
description: Create or update project Cursor rules with mandatory duplicate save to coding-standard and .cursor/rules. Use when adding, editing, or syncing .mdc rules, coding-standard, or .cursor/rules.
paths:
  - "coding-standard/**"
  - ".cursor/rules/**"
  - ".cursor/skills/**"
---

# Write project rules

## Dup save (bắt buộc)

1. Ghi canonical: `coding-standard/<name>.mdc`
2. Copy cùng nội dung: `.cursor/rules/<name>.mdc`
3. Chạy `bash coding-standard/sync-cursor-rules.sh`

Không chỉ sửa một phía.

## Format

`.mdc` + YAML frontmatter. `alwaysApply: true` chỉ cho constraint mọi chat. Ngắn, 1 concern, có Sai/Đúng.

Tham chiếu: `coding-standard/no-shortcut.mdc`, `coding-standard/behavior-no-train-from-input.mdc`
---
