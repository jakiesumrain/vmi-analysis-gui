#ifndef Stage
  #error Stage must point to the staged application
#endif
#ifndef ReleaseVersion
  #error ReleaseVersion is required
#endif

[Setup]
AppId={{90E641A3-7725-4F6A-BB43-88156592E2BC}
AppName=VMI Analysis
AppVersion={#ReleaseVersion}
AppPublisher=VMI Analysis contributors
AppPublisherURL=https://github.com/jakiesumrain/vmi-analysis-gui
DefaultDirName={localappdata}\Programs\VMI Analysis
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\app-{#ReleaseVersion}\runtime\python.exe

[Files]
Source: "{#Stage}\*"; DestDir: "{app}\app-{#ReleaseVersion}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\VMI Analysis"; Filename: "{app}\app-{#ReleaseVersion}\runtime\python.exe"; Parameters: """{app}\app-{#ReleaseVersion}\launch.py"""; WorkingDir: "{app}"
Name: "{userprograms}\VMI Analysis User Manual"; Filename: "{app}\app-{#ReleaseVersion}\manual.pdf"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\vmi-analysis-cache"
Type: filesandordirs; Name: "{app}\vmi-analysis-settings"
