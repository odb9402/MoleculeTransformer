awk -vFS="" '{for(i=1;i<=NF;i++)w[$i]++}END{for(i in w) printf "%s\t%i\n",i,w[i]}' $1
