# Build the container
echo "Generating SIF"
apptainer build noether.sif noether.def

# Move it to the staff umbrella
echo "Copying SIF to Staff Umbrella"
cp noether.sif /tudelft.net/staff-umbrella/noether/noether.sif