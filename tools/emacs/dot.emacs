(global-font-lock-mode)
(set-background-color "NavyBlue")
(set-face-foreground font-lock-function-name-face "Yellow")
(set-foreground-color "White")
(set-cursor-color "Yellow");
(toggle-truncate-lines)
(setq-default c-basic-offset 2)

;;-----------------------------------------------------------------------
;; MELPA
(require 'package) ;; You might already have this line
(add-to-list 'package-archives
         '("melpa" . "http://melpa.org/packages/") t)
(when (< emacs-major-version 24)
;; For important compatibility libraries like cl-lib
(add-to-list 'package-archives '("gnu" . "http://elpa.gnu.org/packages/")))
(package-initialize) ;; You might already have this line

;;-----------------------------------------------------------------------
; rebind keys
(global-set-key [C-tab] 'eme-unbury-buffer)     ; control-tab
(global-set-key [f9] 'do-compile)
(global-set-key [S-f9] 'do-full-compile)
(global-set-key [S-delete] 'delete-region)
(global-set-key [f2] 'do-print)

(setq compile-root "~/kivati/testing/common/memory_slab")

(defun set-compile-root ()
  "Set compile root."
  (interactive)
  (let ((x (read-file-name "Path:")))
    (setq compile-root (file-name-directory x))))

;  (interactive)
;  (let ((x (read-file-name "Enter file name:")))
;    (message "String:%s" arg)))
;;  (setq compile-root arg))))
    
(defun do-compile () "Execute build"
       (interactive)
       (progn (compile (concat (concat "make -C " compile-root) " -j"))))

;        (progn (compile "make -C ~/genode/genode/build.nova32 -j")))

(defun do-full-compile() "Executed full build"
       (interactive)
       (progn (compile (concat (concat "make clean; make -C " compile-root) " -j"))))
;        (progn (compile "make -C ~/genode/genode/build.nova32 clean; make -j")))   

(defun do-print() "Normal PS print"
  (interactive)
  (progn (global-font-lock-mode)(ps-print-buffer)(global-font-lock-mode))
)

(add-to-list 'auto-mode-alist '("\\.h\\'" . c++-mode))


;;-----------------------------------------------------------------------
; gnuplot
;
(add-to-list 'auto-mode-alist '("\\.gplot\\'" . gnuplot-mode))

;;-----------------------------------------------------------------------
; textlint
;
;(add-to-list 'load-path "~/.emacs.d/textlint/")
;(load "textlint.el")

;;-----------------------------------------------------------------------
; doxymacs
;
(autoload 'doxymacs-mode "doxymacs" "Deal with doxygen." t)
(add-hook 'c-mode-common-hook'doxymacs-mode)

;(add-to-list 'auto-mode-alist '("\\.snpl\\'" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.h\\'" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.hpp\\'" . c++-mode))
;(add-to-list 'auto-mode-alist '("\\.str\\'" . stratego-mode))

;;-----------------------------------------------------------------------
; used for iterating buffers
;
(defun eme-unbury-buffer ()
  "Switch to the last (normal) buffer in the buffer list, Should be
inverse of bury-buffer"
  (interactive)
  (let ((bl (reverse (buffer-list))))
    (while bl
        (let* ((buffer (car bl))
        (name (buffer-name buffer)))
                (cond ((string= (substring name 0 1) " ")) ;; ignore hidden buff
                (t (progn (switch-to-buffer buffer) (setq bl nil))))
        (setq bl (cdr bl))))))
; kill scratch


;
; set up printing
;

;;
;; Define postscript print options for NT
;;
;;(require 'ps-print)
;;(setq ps-paper-type 'ps-a4)
;;(setq ps-font-size 6)
;;(setq ps-lpr-command "print")
;;(setq ps-lpr-switches '("/D:\\\\CENTRAL_PRINT\\egaphoto"))
;;(setq ps-lpr-buffer "d:\\psspool.ps")
;;(setq ps-print-color-p nil)        ;; Disable colour emulation on printer

;;
;; Setup postscript print commands for NT and map to keys
;;
(defun nt-ps-print-buffer-with-faces ()
  (interactive)
  (ps-print-buffer-with-faces ps-lpr-buffer)
  (shell-command
   (apply 'concat (append (list ps-lpr-command " ")
                          ps-lpr-switches
                          (list " " ps-lpr-buffer))))
)
(define-key global-map "\C-cb" 'nt-ps-print-buffer-with-faces)

(defun nt-ps-print-region-with-faces ()
  (interactive)
  (ps-print-region-with-faces (mark) (point) ps-lpr-buffer)
  (shell-command
   (apply 'concat (append (list ps-lpr-command " ")
                          ps-lpr-switches
                          (list " " ps-lpr-buffer))))
)
(define-key global-map "\C-cr" 'nt-ps-print-region-with-faces)


;; (setq
;;  helm-gtags-ignore-case t
;;  helm-gtags-auto-update t
;;  helm-gtags-use-input-at-cursor t
;;  helm-gtags-pulse-at-cursor t
;;  helm-gtags-prefix-key "\C-cg"
;;  helm-gtags-suggested-key-mapping t
;;  )

;; (require 'helm-gtags)
;; ;; Enable helm-gtags-mode
;; (add-hook 'dired-mode-hook 'helm-gtags-mode)
;; (add-hook 'eshell-mode-hook 'helm-gtags-mode)
;; (add-hook 'c-mode-hook 'helm-gtags-mode)
;; (add-hook 'c++-mode-hook 'helm-gtags-mode)
;; (add-hook 'asm-mode-hook 'helm-gtags-mode)

;; (define-key helm-gtags-mode-map (kbd "C-c g a") 'helm-gtags-tags-in-this-function)
;; (define-key helm-gtags-mode-map (kbd "C-j") 'helm-gtags-select)
;; (define-key helm-gtags-mode-map (kbd "M-.") 'helm-gtags-dwim)
;; (define-key helm-gtags-mode-map (kbd "M-,") 'helm-gtags-pop-stack)
;; (define-key helm-gtags-mode-map (kbd "C-c <") 'helm-gtags-previous-history)
;; (define-key helm-gtags-mode-map (kbd "C-c >") 'helm-gtags-next-history)

;; (semantic-add-system-include "/usr/include" 'c++-mode)
;; (semantic-add-system-include "/usr/include/python3.5" 'c++-mode)
;; (semantic-add-system-include "/usr/include" 'c-mode)
;; (semantic-add-system-include "/usr/include/efi" 'c++-mode)
;; (semantic-add-system-include "/usr/include/efi" 'c-mode)


;; company mode
(add-hook 'after-init-hook 'global-company-mode)
;(add-hook 'c-mode-common-hook'company-mode)

;; ggtags - install through ELPA
;;
(add-hook 'c-mode-common-hook
          (lambda ()
            (when (derived-mode-p 'c-mode 'c++-mode 'java-mode 'asm-mode)
              (ggtags-mode 1))))

;; (define-key ggtags-mode-map (kbd "C-c g s") 'ggtags-find-other-symbol)
;; (define-key ggtags-mode-map (kbd "C-c g h") 'ggtags-view-tag-history)
;; (define-key ggtags-mode-map (kbd "C-c g r") 'ggtags-find-reference)
;; (define-key ggtags-mode-map (kbd "C-c g f") 'ggtags-find-file)
;; (define-key ggtags-mode-map (kbd "C-c g c") 'ggtags-create-tags)
;; (define-key ggtags-mode-map (kbd "C-c g u") 'ggtags-update-tags)

;; (define-key ggtags-mode-map (kbd "M-,") 'pop-tag-mark)


;;------------------------------------------------------------------------
;; irony mode
;; (add-hook 'c++-mode-hook 'irony-mode)
;; (add-hook 'c-mode-hook 'irony-mode)
;; (add-hook 'objc-mode-hook 'irony-mode)

;; ;; replace the `completion-at-point' and `complete-symbol' bindings in
;; ;; irony-mode's buffers by irony-mode's function
;; (defun my-irony-mode-hook ()
;;   (define-key irony-mode-map [remap completion-at-point]
;;     'irony-completion-at-point-async)
;;   (define-key irony-mode-map [remap complete-symbol]
;;     'irony-completion-at-point-async))
;; (add-hook 'irony-mode-hook 'my-irony-mode-hook)
;; (add-hook 'irony-mode-hook 'irony-cdb-autosetup-compile-options)

;;
;; ctags
;;
(setq path-to-ctags "/usr/bin/ctags")

(defun create-tags (dir-name)
    "Create tags file."
    (interactive "Directory: ")
    (shell-command
     (format "ctags -f %s -R %s" path-to-ctags (directory-file-name dir-name)))
  )

;; Options Menu Settings

;; End of Options Menu Settings
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(auto-compression-mode t nil (jka-compr))
 '(case-fold-search t)
 '(company-clang-arguments nil)
 '(current-language-environment "UTF-8")
 '(default-input-method "latin-9-prefix")
 '(font-use-system-font t)
 '(global-font-lock-mode t nil (font-lock))
 '(indent-tabs-mode nil)
 '(inhibit-startup-screen t)
 '(setq indent-tabs-mode)
 '(show-paren-mode t)
 '(tab-stop-list
   (quote
    (2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80)))
 '(tab-width 2))



(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:inherit nil :stipple nil :background "NavyBlue" :foreground "White" :inverse-video nil :box nil :strike-through nil :overline nil :underline nil :slant normal :weight normal :height 135 :width normal :foundry "DAMA" :family "Ubuntu Mono")))))



;; TEMPORARY install el-get
;;
;; (url-retrieve
;;   "https://raw.githubusercontent.com/dimitri/el-get/master/el-get-install.el"
;;   (lambda (s)
;;     (goto-char (point-max))
;;     (eval-print-last-sexp)))

