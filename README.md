1. Tools used
   tailscale
   pytorch
   flower - ai

server :
terminal 1: flower-superlink --insecure
terminal 2: flwr run . distributed-setup --stream

client:
terminal 1: flower-supernode --insecure --superlink 100.101.155.80:9092 --node-config "partition-id=0 num-partitions=1"
( here ip can change check using tailscale)
