# Model Card: Privacy-Aware Encrypted Traffic Classifiers

## Model Details
- **Architecture**: Supervised Ensembles (Random Forest, XGBoost) and Linear Baselines (Logistic Regression, Linear SVM).
- **Primary Task**: Multi-class Encrypted Application Traffic Classification (File Transfer, Streaming, VoIP/Chat, Remote Access) and Binary VPN Detection (`vpn` vs. `nonvpn`).
- **Feature Space**: Statistical non-payload metadata and Discrete Wavelet Transform (DWT) coefficients. Payloads, IP/MAC addresses, and ephemeral ports are excluded.
- **Library Versions**: Python 3.13, `scikit-learn` 1.9.0, `xgboost` 3.4.1, `shap` 0.54.1, `pandas` 3.0.2.

## Intended Use
- **Primary Intended Uses**: Network capacity planning, QoS policy enforcement, enterprise security monitoring, and research into metadata privacy-utility trade-offs.
- **Out-of-Scope Uses**: Inline deep packet interception, user identity de-anonymization, differential privacy guarantees, or unauthorized surveillance.

## Training Data & Preprocessing
- **Source**: MIT Lincoln Laboratory VNAT Dataset (Jorgensen et al., IEEE TAI 2024).
- **Sample Distribution**: 178,663 total flows.
- **Leakage Prevention**: Stratified train/test splits (80/20). Feature scaling (`StandardScaler`) and label encoding are fitted strictly on the training partition.
- **Metadata Minimization Configurations**:
  - **Configuration A (43 features)**: Full statistical flow metadata + DWT wavelet sub-bands.
  - **Configuration B (29 features)**: Classical flow statistics; fine-grained wavelets excluded.
  - **Configuration C (10 features)**: Ultra-minimal summary metrics (mean packet sizes, inter-arrival times, size spread).

## Performance Summary

### Multi-Class Application Task (25,000 Flows)
| Model | Config | Features | Accuracy | Macro-F1 | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **B** | 29 | **99.88%** | **0.9941** | 0.0025 ms/flow |
| **XGBoost** | **C** | 10 | **99.74%** | **0.9879** | 0.0020 ms/flow |
| **Random Forest** | **B** | 29 | **99.70%** | **0.9903** | 0.0088 ms/flow |
| **Random Forest** | **C** | 10 | **99.54%** | **0.9842** | 0.0084 ms/flow |

### Binary VPN vs. Non-VPN Task (25,000 Flows)
| Model | Config | Features | Accuracy | Macro-F1 | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **A, B, C** | 43 / 29 / 10 | **99.98%** | **0.9998** | <0.001 ms/flow |
| **Random Forest** | **A, B, C** | 43 / 29 / 10 | **99.98%** | **0.9998** | ~0.008 ms/flow |

## Explainability & Feature Drivers
- **Key Discriminative Features**: Root mean square packet size (`Forward_rms`), backward inter-arrival time (`BIAT`), forward inter-arrival time (`FIAT`), median backward packet size (`Backward_med`), and packet size variance (`Forward_std_dev`).
- **Explainability Technique**: SHAP TreeExplainer and Gini importance ranking confirm that flow burstiness and bidirectional size asymmetry are the primary drivers of application distinction.

## Limitations & Ethical Considerations
- **Non-Payload Is Not Risk-Free**: Packet size distributions and inter-arrival timing can still leak behavioral signatures.
- **Simulation vs. Capture**: Cisco Packet Tracer demonstrates network architecture and routing, but does not provide PCAP packet exports for model ingestion. Wireshark captures must be collected from actual host/VM interfaces.

