#!/bin/bash
# Generate self-signed certificates for mTLS between Flower server and clients
# For production, use a proper CA (e.g., Let's Encrypt, internal PKI)

set -e

CERT_DIR="$(dirname "$0")"
cd "$CERT_DIR"

# Configuration
DAYS=365
COUNTRY="VN"
STATE="HCM"
LOCALITY="Ho Chi Minh City"
ORG="FL Malware Detection"
CN_CA="FL Root CA"
CN_SERVER="FL Server"

echo "=== Generating mTLS Certificates for Federated Learning ==="

# 1. Generate CA (Certificate Authority)
echo "[1/4] Generating CA certificate..."
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days $DAYS -key ca.key -out ca.crt \
    -subj "/C=$COUNTRY/ST=$STATE/L=$LOCALITY/O=$ORG/CN=$CN_CA"

# 2. Generate Server certificate
echo "[2/4] Generating Server certificate..."
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
    -subj "/C=$COUNTRY/ST=$STATE/L=$LOCALITY/O=$ORG/CN=$CN_SERVER"

# Create server extensions file for SAN (Subject Alternative Name)
cat > server_ext.cnf << EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = server
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
EOF

openssl x509 -req -days $DAYS -in server.csr -CA ca.crt -CAkey ca.key \
    -CAcreateserial -out server.crt -extfile server_ext.cnf -extensions v3_req

# 3. Generate Client certificates (for 2 clients by default)
NUM_CLIENTS=${1:-2}
echo "[3/4] Generating $NUM_CLIENTS client certificates..."

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CN_CLIENT="FL Client $i"
    openssl genrsa -out "client_$i.key" 2048
    openssl req -new -key "client_$i.key" -out "client_$i.csr" \
        -subj "/C=$COUNTRY/ST=$STATE/L=$LOCALITY/O=$ORG/CN=$CN_CLIENT"
    
    # Create client extensions file
    cat > "client_${i}_ext.cnf" << EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature
extendedKeyUsage = clientAuth
EOF
    
    openssl x509 -req -days $DAYS -in "client_$i.csr" -CA ca.crt -CAkey ca.key \
        -CAcreateserial -out "client_$i.crt" -extfile "client_${i}_ext.cnf" -extensions v3_req
    
    echo "  - Generated client_$i.key and client_$i.crt"
done

# 4. Cleanup CSR files
echo "[4/4] Cleaning up temporary files..."
rm -f *.csr *.cnf ca.srl

echo ""
echo "=== Certificate Generation Complete ==="
echo ""
echo "Files generated:"
echo "  CA:     ca.crt, ca.key"
echo "  Server: server.crt, server.key"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    echo "  Client $i: client_$i.crt, client_$i.key"
done
echo ""
echo "Usage:"
echo "  Server: python server.py --ssl-certfile certs/server.crt --ssl-keyfile certs/server.key --ssl-ca-certfile certs/ca.crt"
echo "  Client: python client.py --ssl-ca-certfile certs/ca.crt --cid 0"
