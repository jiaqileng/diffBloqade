for N in 4 6 8 10
do 
	for lr in 1e-1 5e-2 1e-2
    	do
		julia ghz.jl $N $lr 
	done
done

    
