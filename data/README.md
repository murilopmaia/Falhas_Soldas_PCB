# Dados

Esta pasta é **ignorada pelo Git** (ver `.gitignore`), exceto este arquivo. O conteúdo ocupa ~400 MB e é reprodutível.

```
data/
├── raw/
│   └── Dataset_Soldas/          # 7.920 recortes 40x40 px, 1.584 por classe
│       ├── Soldas_ausente/
│       ├── Soldas_boas/
│       ├── Soldas_excesso/
│       ├── Soldas_ponte/
│       └── Soldas_pouca/
└── synthetic/                   # gerado por src/generate_boards.py
    ├── images/                  # placas sintéticas (PNG)
    ├── annotations/
    │   ├── yolo/                # <classe> <x_c> <y_c> <w> <h> (normalizado)
    │   └── pascal_voc/          # XML usado pelo SSD / TF Object Detection API
    └── reports/                 # contagem de soldas por classe, por placa
```

## `raw/Dataset_Soldas`

Banco de imagens de soldas de PCB construído nos trabalhos de Mar, Yarlagadda & Fookes (2011) e
Cabral Júnior (2022, dissertação PPGMC/UESC). **Não é de minha autoria e não é redistribuído neste
repositório**: baixe/solicite com a fonte original e extraia mantendo a estrutura acima.

## `synthetic/`

Gere com:

```bash
python src/generate_boards.py   # opção 2 → colunas: 10, linhas: 5, quantidade: 2000
```

Para ver o formato sem baixar nada, há 3 placas de exemplo em [`../samples`](../samples).

> Observação: o campo `<path>` dos XML gerados em 2024 aponta para caminhos absolutos da máquina
> original (`C:\Users\Murilo\...`). Ao treinar, as imagens ficam na mesma pasta dos XML, então esse campo provavelmente não é usado; se der erro, basta regerar as placas.
