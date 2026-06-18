# RunPod SSH Access

Date added: 2026-06-17

RunPod SSH command:

```bash
ssh wrlfy1c9urxgai-64410d04@ssh.runpod.io -i /work/.ssh/runpod_acm_ed25519
```

Exposed TCP SSH command:

```bash
ssh root@69.19.136.173 -p 48547 -i /work/.ssh/runpod_acm_ed25519
```

Use the exposed TCP SSH route for direct `rsync`/`scp` commands that need an
explicit host and port.
