"""SAM 3 video-tracker fine-tuning on CVF-2026 smoke masklets.

One concern per module. The stdlib-only half (`boxes`, `crops`, `masklets`,
`clips`, `config`) is deliberately torch-free so the no-GPU probes and
`config_all --check` stay a second rather than an import of CUDA.
"""
