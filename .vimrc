syntax on

set noerrorbells
set tabstop=4 softtabstop=4
set shiftwidth=4
set expandtab
set smartindent
set nu
set nowrap
set smartcase
set noswapfile
set nobackup
set undodir=~/.vim/undodir
set undofile
set incsearch
set backspace=indent,eol,start
set colorcolumn=172
highlight ColorColumn ctermbg=0 guibg=lightgrey

inoremap jj <Esc>

function! CompileRun()
    if expand('%:e') ==? "py"
        return ":!python3.8 " . expand("%:t")
    elseif expand('%:e') ==? "cpp"
        return ":!g++ -std=c++17 " . expand("%:t") . " -o " . expand("%:t:r") . " && ./" . expand("%:t:r")
    else
        return ""
    endif
endfunction 

nnoremap <expr> mm CompileRun()

call plug#begin('~/.vim/plugged')

Plug 'https://github.com/ycm-core/YouCompleteMe'
Plug 'mbbill/undotree'
Plug 'https://github.com/morhetz/gruvbox'

call plug#end()

set bg=dark
let g:gruvbox_contrast_dark='hard'
colo gruvbox
