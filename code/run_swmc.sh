echo "Sending batch..."
bsub -n 48 "python3 swmc.py"
echo "Done"