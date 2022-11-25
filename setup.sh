# create directory 
mkdir -p ~/.streamlit/

# create config file for streamlit
echo "\
[server]\n\
port = $PORT\n\
enabledCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml