# Dataset Audit Report: MIT Lincoln Laboratory VNAT Dataset

## 1. Dataset Overview
- **Dataset Name**: VPN/Non-VPN Network Application Traffic Dataset (VNAT)
- **Primary Source**: MIT Lincoln Laboratory
- **Reference Paper**: S. Jorgensen et al., *"Extensible Machine Learning for Encrypted Network Traffic Application Labeling via Uncertainty Quantification,"* IEEE Transactions on Artificial Intelligence, vol. 5, no. 1, pp. 420-433, Jan. 2024.
- **Analyzed File**: `archive/ll_mit_tw64_complete.csv`
- **Total Records**: 178,663 flows
- **Total Columns**: 44 (43 statistical/wavelet input features + 1 label column)

## 2. Data Quality & Hygiene Check
- **Missing / Null Values**: 0 (0.00%)
- **Infinite Values (Inf / -Inf)**: 0 (0.00%)
- **Direct Identifiers (IP, MAC, Port, Payload)**: Pre-filtered in this feature matrix. No payload or raw packet byte fields exist.
- **Target Leakage**: No columns contain label-derived indices or timestamps acting as proxies.

## 3. Class Distribution

### Binary Task: VPN vs. Non-VPN
| Class | Flow Count | Percentage |
| :--- | :--- | :--- |
| **nonvpn** | 107,648 | 60.25% |
| **vpn** | 71,015 | 39.75% |
| **Total** | 178,663 | 100.00% |

### Multi-Class Task: Application Category
| Category | Applications Included | Flow Count | Percentage |
| :--- | :--- | :--- | :--- |
| **File Transfer** | `scp`, `sftp`, `rsync` | 159,867 | 89.48% |
| **Streaming** | `vimeo`, `netflix`, `youtube` | 8,360 | 4.68% |
| **VoIP / Chat** | `voip`, `skype-chat` | 6,586 | 3.69% |
| **Remote Access** | `rdp`, `ssh` | 3,850 | 2.15% |
| **Total** | 10 applications | 178,663 | 100.00% |

### Granular 20-Class Distribution
| Class Label | Count | Class Label | Count |
| :--- | :--- | :--- | :--- |
| `nonvpn_scp` | 69,528 | `vpn_rdp` | 1,699 |
| `vpn_rsync` | 27,135 | `vpn_vimeo` | 1,614 |
| `nonvpn_sftp` | 22,194 | `nonvpn_skype-chat` | 1,503 |
| `vpn_sftp` | 18,636 | `vpn_voip` | 1,356 |
| `vpn_scp` | 16,594 | `nonvpn_rdp` | 1,351 |
| `nonvpn_rsync` | 5,780 | `vpn_netflix` | 956 |
| `nonvpn_netflix` | 2,156 | `nonvpn_youtube` | 908 |
| `nonvpn_voip` | 1,979 | `vpn_youtube` | 835 |
| `nonvpn_vimeo` | 1,891 | `vpn_ssh` | 442 |
| `vpn_skype-chat` | 1,748 | `nonvpn_ssh` | 358 |

## 4. Evaluative Notes
Because application categories have class imbalance (File Transfer accounts for ~89% of traffic, while Remote Access accounts for ~2%), standard accuracy would be misleading. **Macro-F1** and per-class recall/precision are required primary metrics to ensure high fidelity across all application categories.

