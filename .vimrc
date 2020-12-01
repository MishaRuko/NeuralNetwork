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

inoremap jj <Esc>

set backspace=indent,eol,start

set colorcolumn=172
highlight ColorColumn ctermbg=0 guibg=lightgrey

call plug#begin('~/.vim/plugged')

Plug 'https://github.com/ycm-core/YouCompleteMe'
Plug 'mbbill/undotree'
Plug 'https://github.com/morhetz/gruvbox'

call plug#end()

set bg=dark
let g:gruvbox_contrast_dark='hard'
colo gruvbox

