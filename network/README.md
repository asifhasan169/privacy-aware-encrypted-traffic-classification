# TechCorp Network Topology Guide (Cisco Packet Tracer)

This directory contains the network configuration scripts, addressing schemes, and testing instructions for the enterprise network demonstration corresponding to Workstream 2.

---

## 1. Enterprise Scenario & Architecture
The network models a mid-sized enterprise, **TechCorp**, partitioned into dedicated functional VLANs to enforce least-privilege segmentation and simulate inter-VLAN encrypted traffic paths (HTTPS web traffic, SCP/SFTP file transfers, and remote management):

```
                   +------------------------+
                   |  R1-TechCorp-Core      |
                   |  (Cisco 2911 Router)   |
                   +-----------+------------+
                               | Gi0/0 (802.1Q Dot1Q Trunk)
                               |
                               | Gi0/1 (Trunk)
                   +-----------+------------+
                   |  SW1-TechCorp-Core     |
                   |  (Cisco 2960 Switch)   |
                   +--+---+---+---+---+---+--+
                      |   |   |   |   |   |
          Fa0/10 -----+   |   |   |   |   +----- Fa0/6
         (VLAN 10)        |   |   |   |         (VLAN 60)
            |             |   |   |   |             |
        [PC-HR]           |   |   |   |         [PC-MON]
                          |   |   |   |   (Traffic Analyst Station)
              Fa0/20 -----+   |   |   +----- Fa0/5
             (VLAN 20)        |   |         (VLAN 50)
                |             |   |             |
            [PC-FIN]          |   |         [Server0]
                              |   |    (Internal Web/Services)
                  Fa0/3 ------+   +----- Fa0/4
                 (VLAN 30)              (VLAN 40)
                    |                       |
                [PC-IT]                 [PC-GUEST]
```

---

## 2. IP Addressing & VLAN Table

| Department / Role | VLAN ID | Network Subnet | Default Gateway | Host Device | Host IP Address | Switch Port |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HR** | 10 | `192.168.10.0/24` | `192.168.10.1` | PC-HR | `192.168.10.10` | `Fa0/10` |
| **Finance** | 20 | `192.168.20.0/24` | `192.168.20.1` | PC-FIN | `192.168.20.10` | `Fa0/20` |
| **IT Support** | 30 | `192.168.30.0/24` | `192.168.30.1` | PC-IT | `192.168.30.10` | `Fa0/3` |
| **Guest** | 40 | `192.168.40.0/24` | `192.168.40.1` | PC-GUEST | `192.168.40.10` | `Fa0/4` |
| **Server Farm** | 50 | `192.168.50.0/24` | `192.168.50.1` | Server0 | `192.168.50.10` | `Fa0/5` |
| **Monitoring** | 60 | `192.168.60.0/24` | `192.168.60.1` | PC-MON | `192.168.60.10` | `Fa0/6` |

---

## 3. Implementation Steps in Cisco Packet Tracer

1. **Place Devices:**
   - 1x Router: Cisco 2911 (`R1-TechCorp-Core`)
   - 1x Switch: Cisco Catalyst 2960 (`SW1-TechCorp-Core`)
   - 1x Server: `Server-PT` (`Server0`)
   - 5x PCs: `PC-HR`, `PC-FIN`, `PC-IT`, `PC-GUEST`, `PC-MON`
2. **Cables & Connections (Copper Straight-Through):**
   - Connect `R1` (`GigabitEthernet0/0`) to `SW1` (`GigabitEthernet0/1`).
   - Connect PC/Server endpoints to their assigned switch ports as listed in the table above.
3. **Configure Devices:**
   - Paste `device_configs/switch_2960_config.txt` into SW1 CLI.
   - Paste `device_configs/router_2911_config.txt` into R1 CLI.
   - Assign static IP, subnet mask (`255.255.255.0`), and gateway on each PC/Server Desktop tab.
4. **Verification Commands:**
   - On SW1: `show vlan brief`, `show interfaces trunk`
   - On R1: `show ip interface brief`, `show ip route`
   - On PC-HR:
     - Ping default gateway: `ping 192.168.10.1`
     - Inter-VLAN ping to Server0: `ping 192.168.50.10`
     - Inter-VLAN ping to PC-FIN: `ping 192.168.20.10`
5. **Simulation Mode Demonstration:**
   - Switch Packet Tracer to **Simulation Mode**.
   - Send an HTTP/HTTPS or ICMP PDU from `PC-HR` to `Server0`.
   - Observe the packet route up the 802.1Q trunk link to R1 subinterface `g0/0.10`, routing across to `g0/0.50`, down the trunk to SW1, and delivery to `Server0`.

---

## 4. Methodological Distinction: Packet Tracer vs. Wireshark
* **Packet Tracer** validates network architecture, VLAN segmentation, and routing behavior visually.
* **Wireshark** is executed on the researcher's local machine or authorized VM to inspect real TLS/HTTPS packets, demonstrating the opacity of encrypted payloads and the visibility of flow-level metadata.

