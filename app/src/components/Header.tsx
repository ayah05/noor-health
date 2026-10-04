import {
  HeartPulse,
  WifiOff,
} from "lucide-react";

export default function Header() {
  return (
    <header className="header">
      <div className="brand">
        <div className="brandIcon">
          <HeartPulse size={24} />
        </div>

        <div>
          <h1>Noor Health</h1>
          <p>Clinical intake assistant</p>
        </div>
      </div>

      <div className="offlineBadge">
        <WifiOff size={16} />
        Offline-first
      </div>
    </header>
  );
}