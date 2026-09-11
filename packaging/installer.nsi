; ─────────────────────────────────────────────────────────────────────────────
; AlexGPT installer (NSIS 3.x, Modern UI 2). Собирается из build.ps1:
;   makensis /DAPP_VERSION=1.1.0 installer.nsi
;
;   • Ставит в %LOCALAPPDATA%\Programs\AlexGPT без UAC,
;     или в Program Files, если установщик запущен от администратора
;   • Ярлыки: рабочий стол + меню Пуск, запись в "Установка и удаление программ"
;   • Настройки и чаты лежат в %LOCALAPPDATA%\AlexGPT и при удалении остаются
; ─────────────────────────────────────────────────────────────────────────────

Unicode true
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!ifndef APP_VERSION
  !define APP_VERSION "1.1.0"
!endif
!define APP_NAME      "AlexGPT"
!define APP_PUBLISHER "ALEXaloysha"
!define APP_EXE       "AlexGPT.exe"
!define APP_REGKEY    "Software\${APP_NAME}"
!define UNINST_KEY    "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

Name        "${APP_NAME} ${APP_VERSION}"
OutFile     "dist\AlexGPT-${APP_VERSION}-setup.exe"
BrandingText "${APP_NAME} ${APP_VERSION}"

InstallDir   "$LOCALAPPDATA\Programs\${APP_NAME}"
InstallDirRegKey HKCU "${APP_REGKEY}" "InstallDir"

RequestExecutionLevel user
ShowInstDetails  show
ShowUnInstDetails show

!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON   "alexgpt.ico"
!define MUI_UNICON "alexgpt.ico"

!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Запустить ${APP_NAME}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "Russian"
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  UserInfo::GetAccountType
  Pop $0
  ${If} $0 == "Admin"
    SetShellVarContext all
    StrCpy $INSTDIR "$PROGRAMFILES64\${APP_NAME}"
  ${Else}
    SetShellVarContext current
  ${EndIf}

  System::Call 'kernel32::CreateMutex(p 0, i 0, t "AlexGPTInstaller") p .r1 ?e'
  Pop $0
  StrCmp $0 0 +3
    MessageBox MB_OK|MB_ICONEXCLAMATION "Установщик ${APP_NAME} уже запущен."
    Abort
FunctionEnd

Function un.onInit
  UserInfo::GetAccountType
  Pop $0
  ${If} $0 == "Admin"
    SetShellVarContext all
  ${Else}
    SetShellVarContext current
  ${EndIf}
FunctionEnd

Section "AlexGPT (обязательно)" SecMain
  SectionIn RO

  ; /T закрывает и дочерние процессы (сервер, QtWebEngineProcess) только этого приложения
  nsExec::ExecToLog 'taskkill /F /IM ${APP_EXE} /T'
  Sleep 500

  ; Файлы прошлой версии не должны смешиваться с новыми
  RMDir /r "$INSTDIR\_internal"

  SetOutPath "$INSTDIR"
  File /r "dist\AlexGPT\*.*"

  WriteRegStr   HKCU "${APP_REGKEY}" "InstallDir"  "$INSTDIR"
  WriteRegStr   HKCU "${APP_REGKEY}" "Version"     "${APP_VERSION}"

  WriteRegStr   SHCTX "${UNINST_KEY}" "DisplayName"     "${APP_NAME}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "DisplayVersion"  "${APP_VERSION}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "Publisher"       "${APP_PUBLISHER}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr   SHCTX "${UNINST_KEY}" "DisplayIcon"     "$INSTDIR\${APP_EXE}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr   SHCTX "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoModify"   1
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoRepair"   1

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD SHCTX "${UNINST_KEY}" "EstimatedSize" $0

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Ярлыки на рабочем столе и в меню Пуск" SecShortcuts
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"   "$INSTDIR\${APP_EXE}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\Удалить.lnk"        "$INSTDIR\Uninstall.exe"
  CreateShortcut  "$DESKTOP\${APP_NAME}.lnk"                   "$INSTDIR\${APP_EXE}"
SectionEnd

LangString DESC_SecMain      ${LANG_RUSSIAN} "Само приложение AlexGPT (обязательно)."
LangString DESC_SecShortcuts ${LANG_RUSSIAN} "Создать ярлыки на рабочем столе и в меню Пуск."
LangString DESC_SecMain      ${LANG_ENGLISH} "AlexGPT application (required)."
LangString DESC_SecShortcuts ${LANG_ENGLISH} "Create desktop and Start menu shortcuts."

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain}      $(DESC_SecMain)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecShortcuts} $(DESC_SecShortcuts)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  nsExec::ExecToLog 'taskkill /F /IM ${APP_EXE} /T'
  Sleep 500

  Delete "$DESKTOP\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Удалить.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"

  Delete   "$INSTDIR\${APP_EXE}"
  Delete   "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR\_internal"
  RMDir    "$INSTDIR"

  DeleteRegKey HKCU "${APP_REGKEY}"
  DeleteRegKey SHCTX "${UNINST_KEY}"
SectionEnd
