@echo off
rem Minimal makefile for Sphinx documentation on Windows

if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build
if "%SPHINXOPTS%" == "" set SPHINXOPTS=
if "%SOURCEDIR%" == "" set SOURCEDIR=.
if "%BUILDDIR%" == "" set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "livehtml" goto livehtml
if "%1" == "pdf" goto pdf
if "%1" == "epub" goto epub

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
goto end

:html
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:livehtml
cd /d "%~dp0"
sphinx-autobuild -b html %SPHINXOPTS% %SOURCEDIR% %BUILDDIR%\html
goto end

:pdf
%SPHINXBUILD% -M latexpdf %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:epub
%SPHINXBUILD% -M epub %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:end
